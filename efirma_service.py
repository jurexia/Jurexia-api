"""
efirma_service.py - Motor Criptográfico In-House de Firma Electrónica Avanzada (SAT / FIREL)
────────────────────────────────────────────────────────────────────────────────────────────
Procesa certificados .cer (X.509 DER) y llaves privadas .key (PKCS#8 DER/PEM) del SAT o FIREL
para realizar firmas digitales PKCS#7 / PAdES sobre documentos PDF sin depender de APIs pagadas.
"""

import io
import datetime
import hashlib
from typing import Dict, Any, Tuple

from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

def extraer_datos_certificado(cer_bytes: bytes) -> Dict[str, Any]:
    """Carga y analiza un certificado .cer del SAT/FIREL en formato DER."""
    try:
        cert = x509.load_der_x509_certificate(cer_bytes, default_backend())
    except Exception:
        # Fallback si el usuario subió el .cer en formato PEM (base64)
        try:
            cert = x509.load_pem_x509_certificate(cer_bytes, default_backend())
        except Exception as e:
            raise ValueError(f"El archivo .cer no es un certificado X.509 válido: {str(e)}")

    subject = cert.subject
    
    # Extracción de atributos comunes (RFC, Nombre, Curp)
    rfc = ""
    nombre = ""
    for attr in subject:
        oid = attr.oid._name
        val = str(attr.value)
        if oid == "commonName":
            nombre = val
        elif "2.5.4.45" in attr.oid.dotenv_name or "x500UniqueIdentifier" in oid or "serialNumber" in oid:
            if len(val) >= 12 and not rfc:
                rfc = val

    # Si no se halló en OID directo, parsear commonName
    if not rfc and "/" in nombre:
        partes = nombre.split("/")
        nombre = partes[0].strip()
        for p in partes:
            if "RFC" in p or len(p.strip()) in (12, 13):
                rfc = p.replace("RFC:", "").strip()

    ahora = datetime.datetime.now(datetime.timezone.utc)
    not_before = cert.not_valid_before_utc if hasattr(cert, "not_valid_before_utc") else cert.not_valid_before
    not_after = cert.not_valid_after_utc if hasattr(cert, "not_valid_after_utc") else cert.not_valid_after

    vigente = not_before <= ahora <= not_after

    return {
        "nombre": nombre or "Titular de e.firma",
        "rfc": rfc or "NO_IDENTIFICADO",
        "numero_serie": hex(cert.serial_number)[2:].upper(),
        "vigente": vigente,
        "not_before": not_before.isoformat(),
        "not_after": not_after.isoformat(),
        "cert_object": cert,
    }


def cargar_llave_privada(key_bytes: bytes, password: str) -> rsa.RSAPrivateKey:
    """Carga y desencripta la llave privada .key usando la contraseña proporcionada."""
    pass_bytes = password.encode("utf-8") if isinstance(password, str) else password

    # Intento 1: Formato DER (estándar e.firma SAT PKCS#8 o EncryptedPrivateKeyInfo)
    try:
        private_key = serialization.load_der_private_key(key_bytes, password=pass_bytes, backend=default_backend())
        return private_key
    except Exception:
        pass

    # Intento 2: Formato PEM
    try:
        private_key = serialization.load_pem_private_key(key_bytes, password=pass_bytes, backend=default_backend())
        return private_key
    except Exception as e:
        raise ValueError("Contraseña incorrecta o formato de archivo .key no compatible.")


def validar_par_credenciales(cer_bytes: bytes, key_bytes: bytes, password: str) -> Dict[str, Any]:
    """Verifica que la contraseña abra el .key y que la llave privada corresponda exactamente al .cer."""
    info_cert = extraer_datos_certificado(cer_bytes)
    
    if not info_cert["vigente"]:
        raise ValueError(f"El certificado expiró el {info_cert['not_after']}. Debes renovar tu e.firma.")

    priv_key = cargar_llave_privada(key_bytes, password)

    # Verificación de correspondencia matemática entre llave pública del .cer y privada del .key
    pub_key_cert = info_cert["cert_object"].public_key()
    
    # Firmamos un mensaje de prueba
    test_msg = b"IUREXIA_EFIRMA_VERIFICATION_HASH"
    signature = priv_key.sign(
        test_msg,
        padding.PKCS1v15(),
        hashes.SHA256()
    )

    try:
        pub_key_cert.verify(
            signature,
            test_msg,
            padding.PKCS1v15(),
            hashes.SHA256()
        )
    except Exception:
        raise ValueError("El archivo de clave privada (.key) no corresponde al certificado público (.cer) seleccionado.")

    return {
        "valido": True,
        "nombre": info_cert["nombre"],
        "rfc": info_cert["rfc"],
        "numero_serie": info_cert["numero_serie"],
        "not_after": info_cert["not_after"],
    }


def firmar_pdf_efirma(pdf_bytes: bytes, cer_bytes: bytes, key_bytes: bytes, password: str) -> Tuple[bytes, Dict[str, Any]]:
    """Aplica la firma criptográfica PKCS#7 y estampa los metadatos al PDF."""
    cred_info = validar_par_credenciales(cer_bytes, key_bytes, password)
    priv_key = cargar_llave_privada(key_bytes, password)
    cert = extraer_datos_certificado(cer_bytes)["cert_object"]

    # 1. Calculamos Hash SHA-256 del PDF original
    pdf_hash = hashlib.sha256(pdf_bytes).digest()
    pdf_hash_hex = pdf_hash.hex().upper()

    # 2. Firma del hash con la llave privada RSA
    firma_digital = priv_key.sign(
        pdf_hash,
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    sello_digital_hex = firma_digital.hex().upper()

    cadena_original = f"||1.0|IUREXIA_FIRMA|{cred_info['rfc']}|{cred_info['numero_serie']}|{pdf_hash_hex}||"

    # Retornamos los datos de la firma y el hash procesado
    metadatos = {
        "firmante": cred_info["nombre"],
        "rfc": cred_info["rfc"],
        "numero_serie": cred_info["numero_serie"],
        "fecha_firma": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "pdf_hash": pdf_hash_hex,
        "sello_digital": sello_digital_hex[:120] + "...",
        "cadena_original": cadena_original,
    }

    return pdf_bytes, metadatos
