# Speech-to-text

Tensorstead serves speech-to-text through the **vLLM** runtime it already
ships. vLLM exposes the OpenAI-compatible `POST /v1/audio/transcriptions` and
`/v1/audio/translations` endpoints for speech models, including Whisper,
Voxtral, Qwen3-ASR, Granite Speech and Parakeet. You don't need a separate
runtime. The deployment is acquired, recorded, started, observed and restored
like any other.

## Why vLLM and not a dedicated Whisper server

The dedicated servers were checked against their source in October 2026 and
rejected for this product:

- **whisper.cpp's server** has no authentication. It has a `/load` route that
  swaps the served model by file path, and its CUDA image is published for
  x86_64 only.
- **speaches** (formerly faster-whisper-server) downloads models itself and
  chooses one per request, so a deployment record could not say what is being
  served. On ARM nodes such as a DGX Spark, its `ctranslate2` engine has no CUDA
  build, so Whisper would run on the CPU.

vLLM runs on the GPU on both x86_64 and ARM, reads an inference key from
`VLLM_API_KEY` when you want one, and loads exactly the model you acquired.

## 1. Build a vLLM image with audio support

vLLM's audio dependencies are an optional extra (`vllm[audio]`: `av`, `scipy`,
`soundfile`, `soxr`, `mistral_common[audio]`). The Dockerfile for the stock
image does not install that extra, so build an image that adds it. Use the
vLLM image you already run as the base, pinned by digest:

```sh
stead buildspec set vllm-audio \
  --base 'nvcr.io/nvidia/vllm@sha256:<digest-you-already-run>' \
  --step 'python3 -m pip install --no-cache-dir --root-user-action=ignore av scipy soundfile soxr "mistral_common[audio]"' \
  --step 'python3 -c "import av, scipy, soundfile, soxr"'

stead image build vllm-audio --node spark-01 --reference local/vllm:audio
```

The second step fails the build if any library did not install, so a broken
image never reaches a deployment. Pin package versions in the first step once
you have a combination that works.

## 2. Acquire a speech model

```sh
stead model acquire --source huggingface --id openai/whisper-large-v3-turbo \
  --revision <immutable-commit-sha> --node spark-01
```

Any speech model vLLM supports works the same way. Check vLLM's supported
models list for the image you built.

## 3. Deploy and start it

```sh
stead deployment create --name whisper \
    --model <model-id> --runtime vllm --image local/vllm:audio \
    --node spark-01 --endpoint 10.0.0.11:8001 \
    --config served_model_name=whisper-large-v3-turbo

stead deployment start whisper
stead deployment show whisper
```

`served_model_name` is the name clients pass as `model`. A speech model is
small, so it can share a node with a language model. Give it its own port, and
set `gpu_memory_utilization` so the two together fit in the node's memory.

## 4. Transcribe

```sh
curl http://10.0.0.11:8001/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F model=whisper-large-v3-turbo
```

Any OpenAI client works by pointing its base URL at the endpoint:

```python
from openai import OpenAI

client = OpenAI(base_url="http://10.0.0.11:8001/v1", api_key="unused")
with open("meeting.wav", "rb") as audio:
    print(client.audio.transcriptions.create(model="whisper-large-v3-turbo", file=audio).text)
```

## Inference keys

Optional, as for every deployment. To require one, bind it to the deployment
(`stead inferencekey bind whisper --name <key>`) and restart. Clients then send
`Authorization: Bearer <key>`. See [operations.md](./operations.md#inference-credentials).

## Not verified yet

This path has been checked against vLLM's source, not yet run on hardware. If a
transcription request fails, `stead deployment runtime whisper` shows vLLM's own
log. A missing audio library appears there by name.
