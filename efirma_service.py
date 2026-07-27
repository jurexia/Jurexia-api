"""
efirma_service.py — Firma electrónica avanzada (e.firma del SAT / FIREL) sobre PDF.

Qué hace: toma el .cer y el .key del abogado, comprueba que van juntos y que la
contraseña abre la llave, y **incrusta una firma PKCS#7 real dentro del PDF**
usando pyHanko. El resultado es un PDF firmado que cualquier visor (Adobe, Foxit,
el validador del SAT) puede verificar de forma independiente.

─── Por qué se reescribió por completo ──────────────────────────────────────

La versión anterior tenía dos fallas que la hacían inservible, y una de ellas
era peligrosa:

1. `extraer_datos_certificado` reventaba con AttributeError en cualquier
   certificado real del SAT, porque leía `attr.oid.dotenv_name`, un atributo
   que no existe en `cryptography` (es `dotted_string`). Cualquier .cer con
   algo más que un commonName —o sea, todos los del SAT— tiraba la petición.

2. `firmar_pdf_efirma` calculaba el sello RSA correctamente… y luego hacía
   `return pdf_bytes`, devolviendo **el PDF original sin tocar**. El usuario
   recibía un archivo idéntico al que subió, con metadatos que decían "firmado".
   Eso es exactamente lo que no se debe hacer: entregar un documento sin firma
   presentándolo como firmado. Un abogado podría haberlo presentado en un
   juzgado creyéndolo válido.

Ahora la firma va dentro del PDF de verdad. Si algo falla, se lanza una
excepción — nunca se devuelve un PDF a medias.

─── Lo que esta firma es y lo que no es ─────────────────────────────────────

ES: una firma electrónica avanzada con el certificado del titular, con validez
jurídica conforme al artículo 97 del Código de Comercio y a la Ley de Firma
Electrónica Avanzada, verificable criptográficamente por un tercero.

NO ES: una firma que Adobe Reader marque con palomita verde de forma automática.
Adobe confía en su propia lista de autoridades (AATL) y el SAT no está en ella,
así que mostrará "validez desconocida" salvo que el verificador agregue la raíz
del SAT a sus certificados de confianza. Esto es una limitación de Adobe, no de
la firma. Conviene decírselo al usuario en vez de que lo descubra en el juzgado.

─── Seguridad ───────────────────────────────────────────────────────────────

La llave privada (.key) y la contraseña viven sólo en memoria durante la
petición: nunca se escriben a disco, nunca se registran en logs y nunca se
guardan en base de datos. pyHanko recibe los objetos ya cargados en RAM.
"""

from __future__ import annotations

import datetime
import hashlib
import io
from typing import Any, Dict, Tuple

from asn1crypto import keys as asn1_keys
from asn1crypto import x509 as asn1_x509
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

# OIDs que el SAT usa para meter el RFC y la CURP del titular en el certificado.
# El RFC va en x500UniqueIdentifier (2.5.4.45) y a veces en serialNumber (2.5.4.5).
OID_X500_UNIQUE_IDENTIFIER = "2.5.4.45"
OID_SERIAL_NUMBER = "2.5.4.5"


class ErrorEfirma(ValueError):
    """Problema atribuible a lo que subió el usuario (archivos o contraseña)."""


# ─── Lectura del certificado ─────────────────────────────────────────────────


def _limpiar_rfc(valor: str) -> str:
    """
    El SAT mete el RFC pegado a la CURP en el mismo campo, separado por " / ".
    Un RFC de persona física trae 13 caracteres y uno moral 12.
    """
    valor = (valor or "").strip()
    if "/" in valor:
        valor = valor.split("/")[0].strip()
    return valor


def extraer_datos_certificado(cer_bytes: bytes) -> Dict[str, Any]:
    """Lee un .cer del SAT o FIREL (DER, o PEM si el usuario lo convirtió)."""
    try:
        cert = x509.load_der_x509_certificate(cer_bytes, default_backend())
    except Exception:
        try:
            cert = x509.load_pem_x509_certificate(cer_bytes, default_backend())
        except Exception as e:
            raise ErrorEfirma(
                f"El archivo .cer no es un certificado X.509 válido: {e}"
            )

    nombre = ""
    rfc = ""
    curp = ""

    for attr in cert.subject:
        # `dotted_string` es el nombre correcto del atributo. Usar el nombre
        # equivocado aquí fue el bug que tiraba todas las peticiones.
        oid_num = attr.oid.dotted_string
        oid_nombre = attr.oid._name or ""
        valor = str(attr.value)

        if oid_nombre == "commonName":
            nombre = valor
        elif oid_num in (OID_X500_UNIQUE_IDENTIFIER, OID_SERIAL_NUMBER):
            candidato = _limpiar_rfc(valor)
            # Nos quedamos con el primero que tenga pinta de RFC (12 o 13).
            if not rfc and 12 <= len(candidato) <= 13:
                rfc = candidato
            # La CURP siempre trae 18 caracteres.
            if len(valor.strip()) == 18:
                curp = valor.strip()
            elif "/" in valor:
                posible_curp = valor.split("/")[-1].strip()
                if len(posible_curp) == 18:
                    curp = posible_curp

    # Algunos certificados viejos meten todo en el commonName separado por "/".
    if not rfc and "/" in nombre:
        partes = [p.strip() for p in nombre.split("/")]
        nombre = partes[0]
        for p in partes[1:]:
            limpio = p.replace("RFC:", "").strip()
            if 12 <= len(limpio) <= 13 and not rfc:
                rfc = limpio

    ahora = datetime.datetime.now(datetime.timezone.utc)
    not_before = getattr(cert, "not_valid_before_utc", None) or cert.not_valid_before
    not_after = getattr(cert, "not_valid_after_utc", None) or cert.not_valid_after
    if not_before.tzinfo is None:
        not_before = not_before.replace(tzinfo=datetime.timezone.utc)
    if not_after.tzinfo is None:
        not_after = not_after.replace(tzinfo=datetime.timezone.utc)

    return {
        "nombre": nombre or "Titular de e.firma",
        "rfc": rfc or "NO_IDENTIFICADO",
        "curp": curp,
        "numero_serie": _numero_serie_legible(cert),
        "vigente": not_before <= ahora <= not_after,
        "not_before": not_before.isoformat(),
        "not_after": not_after.isoformat(),
        "emisor": cert.issuer.rfc4514_string(),
        "cert_object": cert,
    }


def _numero_serie_legible(cert) -> str:
    """
    El SAT imprime el número de serie como texto ASCII (p. ej. "00001000000...")
    y no como el hexadecimal crudo. Se intenta decodificar; si no es ASCII
    imprimible, se cae al hexadecimal.
    """
    crudo = cert.serial_number.to_bytes(
        (cert.serial_number.bit_length() + 7) // 8 or 1, "big"
    )
    try:
        texto = crudo.decode("ascii")
        if texto.isprintable() and texto.strip():
            return texto
    except UnicodeDecodeError:
        pass
    return hex(cert.serial_number)[2:].upper()


# ─── Llave privada ───────────────────────────────────────────────────────────


def cargar_llave_privada(key_bytes: bytes, password: str) -> rsa.RSAPrivateKey:
    """Abre el .key del SAT (PKCS#8 cifrado, en DER) con la contraseña."""
    clave = password.encode("utf-8") if isinstance(password, str) else password

    for cargar in (serialization.load_der_private_key, serialization.load_pem_private_key):
        try:
            return cargar(key_bytes, password=clave, backend=default_backend())
        except Exception:
            continue

    raise ErrorEfirma(
        "No se pudo abrir el archivo .key. Revisa que la contraseña sea la de tu "
        "clave privada (no la del RFC) y que el archivo no esté dañado."
    )


def validar_par_credenciales(cer_bytes: bytes, key_bytes: bytes, password: str) -> Dict[str, Any]:
    """
    Comprueba tres cosas antes de dejar firmar: que el certificado esté vigente,
    que la contraseña abra la llave, y que esa llave sea de ese certificado.
    """
    info = extraer_datos_certificado(cer_bytes)

    if not info["vigente"]:
        raise ErrorEfirma(
            f"El certificado no está vigente (vence el {info['not_after'][:10]}). "
            "Renueva tu e.firma en el SAT."
        )

    llave = cargar_llave_privada(key_bytes, password)

    # Que la llave privada case con la pública del certificado: se firma un
    # mensaje de prueba y se verifica con el certificado.
    prueba = b"IUREXIA_EFIRMA_VERIFICATION"
    try:
        firma = llave.sign(prueba, padding.PKCS1v15(), hashes.SHA256())
        info["cert_object"].public_key().verify(
            firma, prueba, padding.PKCS1v15(), hashes.SHA256()
        )
    except Exception:
        raise ErrorEfirma(
            "El archivo .key no corresponde al .cer que subiste. Asegúrate de que "
            "ambos sean del mismo trámite de e.firma."
        )

    return {
        "valido": True,
        "nombre": info["nombre"],
        "rfc": info["rfc"],
        "curp": info["curp"],
        "numero_serie": info["numero_serie"],
        "not_after": info["not_after"],
        "emisor": info["emisor"],
    }


# ─── Firma del PDF ───────────────────────────────────────────────────────────


def firmar_pdf_efirma(
    pdf_bytes: bytes,
    cer_bytes: bytes,
    key_bytes: bytes,
    password: str,
    razon: str = "Firma electrónica avanzada del titular",
    lugar: str = "México",
) -> Tuple[bytes, Dict[str, Any]]:
    """
    Incrusta una firma PKCS#7 real dentro del PDF y devuelve los bytes firmados.

    Devuelve `(pdf_firmado, metadatos)`. El PDF que sale **no** es el que entró:
    lleva la firma dentro. Si por lo que sea no se pudiera firmar, esto lanza una
    excepción en vez de devolver el original — quien llama nunca debe poder
    entregar un PDF sin firmar creyendo que la lleva.
    """
    from pyhanko.pdf_utils.incremental_writer import IncrementalPdfFileWriter
    from pyhanko.sign import signers
    from pyhanko_certvalidator.registry import SimpleCertificateStore

    if not pdf_bytes:
        raise ErrorEfirma("No se recibió ningún PDF para firmar.")

    credenciales = validar_par_credenciales(cer_bytes, key_bytes, password)
    llave = cargar_llave_privada(key_bytes, password)

    # pyHanko trabaja con asn1crypto, así que se reserializan cert y llave.
    # Todo en memoria: la llave privada jamás toca el disco.
    cert_der = extraer_datos_certificado(cer_bytes)["cert_object"].public_bytes(
        serialization.Encoding.DER
    )
    llave_der = llave.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    firmante = signers.SimpleSigner(
        signing_cert=asn1_x509.Certificate.load(cert_der),
        signing_key=asn1_keys.PrivateKeyInfo.load(llave_der),
        cert_registry=SimpleCertificateStore(),
    )

    try:
        escritor = IncrementalPdfFileWriter(io.BytesIO(pdf_bytes))
        salida = signers.sign_pdf(
            escritor,
            signers.PdfSignatureMetadata(
                field_name="FirmaIurexia",
                reason=razon,
                location=lugar,
                name=credenciales["nombre"],
            ),
            signer=firmante,
        )
        pdf_firmado = salida.getvalue()
    except Exception as e:
        raise ErrorEfirma(
            f"No se pudo incrustar la firma en el PDF: {e}. "
            "Revisa que el archivo sea un PDF válido y no esté protegido con contraseña."
        )

    # Salvaguarda: si el PDF saliera idéntico al que entró, algo falló y no se
    # debe entregar como firmado. Es el error exacto que tenía la versión previa.
    if pdf_firmado == pdf_bytes:
        raise ErrorEfirma("La firma no quedó incrustada en el PDF; no se entrega el archivo.")

    metadatos = {
        "firmante": credenciales["nombre"],
        "rfc": credenciales["rfc"],
        "numero_serie": credenciales["numero_serie"],
        "emisor": credenciales["emisor"],
        "fecha_firma": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        # Huella del documento ya firmado: es la que sirve para cotejar después.
        "hash_documento_firmado": hashlib.sha256(pdf_firmado).hexdigest().upper(),
        "algoritmo": "RSA con SHA-256 (PKCS#7 incrustado en el PDF)",
        "vigencia_certificado": credenciales["not_after"],
        "nota_validacion": (
            "Firma electrónica avanzada verificable criptográficamente. Adobe Reader "
            "puede mostrarla como 'validez desconocida' porque no incluye a la "
            "autoridad del SAT en su lista de confianza; eso no afecta su validez "
            "jurídica ni la verificación ante el SAT."
        ),
    }

    return pdf_firmado, metadatos
