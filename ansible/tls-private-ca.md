# Private CA and Spark Certificate

This is a small, manual private-certificate procedure for an **advanced override** when you prefer to manage certificates yourself. Normal Tensorstead setup creates its private trust automatically; you do not need this guide for a standalone Spark. It remains intentionally separate from Ansible because manually handling a trust authority and private keys deserves deliberate review.

> Security review note: this is a security-focused, reviewable starting point, not a replacement for your organization's security policy or a professional review. Do not use it for an internet-facing public service. If the Spark will be reachable outside your trusted network, use an organization-approved certificate process instead.

## Before you begin

- Pick the stable DNS name that the inventory will use, for example `spark-alpha.internal`.
- Run the CA commands on a trusted administrator machine, **not** on the Spark.
- Keep the CA private key offline after issuing certificates. Anyone with it can create a certificate trusted by your coordinator.
- The commands below need OpenSSL 1.1.1 or newer. They create files locally; they do not contact or change a Spark.

OpenSSL's certificate-request tooling supports adding a DNS name as a subject alternative name, and its verification tooling can check a certificate against both a CA file and an expected hostname. See the [OpenSSL request documentation](https://docs.openssl.org/4.0/man1/openssl-req/) and [verification options](https://docs.openssl.org/4.0/man1/openssl-verification-options/).

## 1. Create the private CA once

On the trusted administrator machine:

```sh
umask 077
mkdir -p ~/tensorstead-private-ca
cd ~/tensorstead-private-ca

openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:4096 -out tensorstead-ca.key
openssl req -x509 -new -sha256 -days 3650 \
  -key tensorstead-ca.key \
  -out tensorstead-ca.pem \
  -subj '/CN=Tensorstead Private CA' \
  -addext 'basicConstraints=critical,CA:TRUE,pathlen:0' \
  -addext 'keyUsage=critical,keyCertSign,cRLSign' \
  -addext 'subjectKeyIdentifier=hash'
```

Protect `tensorstead-ca.key`. Do not copy it to a Spark, commit it, paste it into chat, or store it in Ansible inventory. `tensorstead-ca.pem` is public trust material and becomes the CA bundle used by Tensorstead.

## 2. Issue one agent certificate

Replace `spark-alpha.internal` below with the exact DNS name in your inventory. Run this on the trusted administrator machine:

```sh
export TENSORSTEAD_AGENT_DNS=spark-alpha.internal

openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:3072 -out "${TENSORSTEAD_AGENT_DNS}.key"
openssl req -new \
  -key "${TENSORSTEAD_AGENT_DNS}.key" \
  -out "${TENSORSTEAD_AGENT_DNS}.csr" \
  -subj "/CN=${TENSORSTEAD_AGENT_DNS}"

cat > "${TENSORSTEAD_AGENT_DNS}.ext" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=DNS:${TENSORSTEAD_AGENT_DNS}
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid,issuer
EOF

openssl x509 -req -sha256 -days 397 \
  -in "${TENSORSTEAD_AGENT_DNS}.csr" \
  -CA tensorstead-ca.pem \
  -CAkey tensorstead-ca.key \
  -CAcreateserial \
  -out "${TENSORSTEAD_AGENT_DNS}.crt" \
  -extfile "${TENSORSTEAD_AGENT_DNS}.ext"
```

The certificate's DNS name is the important part. A certificate for `spark-alpha.internal` will not pass hostname verification when Ansible or the coordinator connects as some other name.

## 3. Verify before copying

```sh
openssl verify -CAfile tensorstead-ca.pem \
  -verify_hostname spark-alpha.internal \
  spark-alpha.internal.crt
```

Expected result: `spark-alpha.internal.crt: OK`.

## 4. Install files where the playbook expects them

Using your normal approved administrator method, install:

| File | Location | Permissions |
|---|---|---|
| `tensorstead-ca.pem` | `/etc/tensorstead/ca.pem` on the coordinator and every agent | Public-to-the-host is acceptable; do not modify it casually. |
| `spark-alpha.internal.crt` | `/etc/tensorstead/agent.crt` on that agent | Readable by the agent service. |
| `spark-alpha.internal.key` | `/etc/tensorstead/agent.key` on that agent | Owner-only, typically `root:root` and mode `0600`. |

The agent certificate and private key exist only on the agent. The coordinator needs the CA bundle so it can verify the agent; it does not need the agent private key.

## 5. Record only paths in Ansible

In your local, uncommitted variable file, set only these paths:

```yaml
tensorstead_agent_ca_bundle: /etc/tensorstead/ca.pem
tensorstead_agent_tls_cert: /etc/tensorstead/agent.crt
tensorstead_agent_tls_key: /etc/tensorstead/agent.key
```

Do not put file contents in YAML. Keep `tensorstead-ca.key`, the agent `.key`, and Tensorstead tokens out of Git, chat, and inventory.

## Renewal and loss

Record the certificate expiry date. Before it expires, issue a replacement certificate with the same DNS name, install it through your approved method, then run the normal Ansible deployment when you are ready. If the CA private key is lost, treat the old private CA as no longer maintainable and make a replacement plan. If it may have been copied or exposed, treat it as compromised: create a new CA and replace the CA bundle and all agent certificates.
