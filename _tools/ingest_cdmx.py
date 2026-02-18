#!/usr/bin/env python3
"""
ingest_cdmx.py — Article-Aware Ingestion for CDMX State Laws
═══════════════════════════════════════════════════════════════════════

Downloads official PDFs from data.consejeria.cdmx.gob.mx,
applies article-aware chunking, generates dense embeddings (text-embedding-3-small),
and upserts to the leyes_estatales Qdrant collection.

BM25 sparse vectors are generated AFTER ingestion via /admin/reingest-sparse.

Usage:
    python ingest_queretaro.py                    # Full pipeline
    python ingest_queretaro.py --delete-only      # Only delete existing Querétaro data
    python ingest_queretaro.py --skip-download     # Skip PDF download (use cached)
"""

import asyncio
import hashlib
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import httpx
import pymupdf  # PyMuPDF
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    PointStruct,
    SparseVector,
    NamedSparseVector,
)

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

QDRANT_URL = os.environ.get(
    "QDRANT_URL",
    "https://d6766dbb-cf4c-40a2-a636-78060cc09ccc.us-east4-0.gcp.cloud.qdrant.io",
)
QDRANT_API_KEY = os.environ.get(
    "QDRANT_API_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4hZwbdZT6esMLx7hjHCi79hD5gLpEAVphmuNGYB3A0Y",
)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

COLLECTION = "leyes_estatales"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
ENTIDAD = "CDMX"

# Article-aware chunking parameters
MAX_CHUNK_TOKENS = 1200       # Max tokens per chunk (~4800 chars)
OVERLAP_CHARS = 400           # Overlap between sub-chunks for long articles
MIN_CHUNK_LEN = 50            # Skip chunks shorter than this (noise)

# PDF download directory
PDF_DIR = Path(__file__).parent / "pdfs_cdmx"

# Rate limiting
EMBED_BATCH_SIZE = 50         # OpenAI allows up to 2048 inputs
EMBED_CONCURRENCY = 5         # Concurrent embedding batches
QDRANT_BATCH_SIZE = 50        # Points per upsert batch

# ══════════════════════════════════════════════════════════════════════
# ALL CDMX LAWS
# ══════════════════════════════════════════════════════════════════════

CDMX_BASE = "https://data.consejeria.cdmx.gob.mx/images/leyes"

@dataclass
class LawDef:
    nombre: str
    url: str
    categoria: str  # constitucion, ley, codigo, ley_organica
    tipo_codigo: str = ""  # PENAL, CIVIL, FISCAL, AMBIENTAL, URBANO, etc.

def _infer_tipo_codigo(nombre: str, categoria: str) -> str:
    """Infer tipo_codigo from law name for Da Vinci filtering."""
    n = nombre.lower()
    if "penal" in n:
        return "PENAL"
    if "civil" in n and "procedimiento" not in n:
        return "CIVIL"
    if "procedimientos civiles" in n:
        return "PROCESAL_CIVIL"
    if "fiscal" in n:
        return "FISCAL"
    if "urbano" in n:
        return "URBANO"
    if "ambiental" in n:
        return "AMBIENTAL"
    if "familiar" in n:
        return "FAMILIAR"
    if "electoral" in n:
        return "ELECTORAL"
    if "constitución" in n or "constituc" in n:
        return "CONSTITUCION"
    if "tránsito" in n or "transito" in n:
        return "TRANSITO"
    if "notarial" in n or "notariado" in n:
        return "NOTARIAL"
    if "salud" in n:
        return "SALUD"
    if "educación" in n or "educacion" in n:
        return "EDUCACION"
    if "transparencia" in n:
        return "TRANSPARENCIA"
    if "laboral" in n or "trabajadores" in n:
        return "LABORAL"
    if "hacienda" in n:
        return "HACIENDA"
    if "seguridad" in n:
        return "SEGURIDAD"
    if "derechos humanos" in n:
        return "DERECHOS_HUMANOS"
    return "GENERAL"

def _infer_jurisdiccion(nombre: str) -> str:
    """Infer jurisdiccion (legal matter) from law name."""
    n = nombre.lower()
    if "penal" in n:
        return "penal"
    if "civil" in n:
        return "civil"
    if "familiar" in n:
        return "familiar"
    if "laboral" in n or "trabajadores" in n:
        return "laboral"
    if "mercantil" in n or "comerci" in n:
        return "mercantil"
    if "fiscal" in n or "hacienda" in n or "tributari" in n:
        return "fiscal"
    if "administrativ" in n:
        return "administrativo"
    if "electoral" in n:
        return "electoral"
    if "ambiental" in n:
        return "ambiental"
    if "constitución" in n or "constituc" in n:
        return "constitucional"
    return "general"

# Build the complete law registry
LAWS: list[LawDef] = []

def _build_laws():
    """Build the full list of CDMX laws."""
    # Constitución (1)
    LAWS.append(LawDef(
        nombre="Constitución Política de la Ciudad de México",
        url=f"{CDMX_BASE}/estatutos/CONSTITUCION_POLITICA_DE_LA_CDMX_14.4.pdf",
        categoria="constitucion",
    ))

    # Leyes (~95)
    ley_entries = [
        ("Ley Orgánica del Congreso de la Ciudad de México", "leyes/2025/LEY_ORGANICA_DEL_CONGRESO_DE_LA_CDMX_7.4.pdf"),
        ("Ley que Regula el Uso de Tecnología para la Seguridad Ciudadana de la CDMX", "2025/181225/LEY_QUE_REGULA_EL_USO_DELA_TECNOLOGA_PARA_LA_SEGURIDAD_CIUDADANA_DE_LA_CDMX_2.5.pdf"),
        ("Ley de Protección a la Salud de las Personas No Fumadoras de la Ciudad de México", "2025/2026/LEY_DE_PROTECCION_A_LA_SALUD_DE_LOS_NO_FUMADORES_2.3.pdf"),
        ("Ley de Protección y Fomento al Empleo para la Ciudad de México", "2025/LEY_DE_PROTECCION_Y_FOMENTO_AL_EMPLEO_PARA_LA_CDMX_2.pdf"),
        ("Ley de Educación de la Ciudad de México", "2025/LEY_DE_EDUCACION_DE_LA_CDMX_3.6.pdf"),
        ("Ley de Auditoría y Control Interno de la Administración Pública de la Ciudad de México", "2025/LEY_DE_AUDITORIA_Y_CONTROL_INTERNO_DE_LA_ADMON_PUBLICA_DE_LA_CDMX_3.3.pdf"),
        ("Ley Orgánica del Instituto de Planeación Democrática y Prospectiva de la Ciudad de México", "2025/LEY_%20ORG_DEL_INST_DE_PLANEACION_DEMOCRATICA%20_Y_PROSPECTIVA_DE_LA_CDMX_3.4.pdf"),
        ("Ley Ambiental de la Ciudad de México", "2025/2026/210126/LEY_AMBIENTAL_DE_LA_CDMX_1.2.pdf"),
        ("Ley de Protección de Datos Personales en Posesión de Sujetos Obligados del Distrito Federal", "leyes/LEYDPROTECCIONDEDATOSPERSONALESPARAELDISTRITOFEDERAL.pdf"),
        ("Ley de Acceso de las Mujeres a una Vida Libre de Violencia de la Ciudad de México", "LEY_DE_ACCESO_DE_LAS_MUJERES_A_UNA_VIDA_LIBRE_DE_VIOLENCIA_DE_LA_CDMX_12.4.pdf"),
        ("Ley de los Derechos de Niñas, Niños y Adolescentes de la Ciudad de México", "LEY_DERECHOS_NINAS_NINOS_ADOLESCENTES_CDMX_7.3.pdf"),
        ("Ley para el Reconocimiento y la Atención de las Personas LGBTTTI+ de la Ciudad de México", "LEY_PARA_RECONOCIMIENTO_Y_LA_ATENCION_DE_LAS_PERSONAS_LGBTTTI_DE_LA_CDMX_4.1.pdf"),
        ("Ley para la Celebración de Espectáculos Públicos de la Ciudad de México", "Ley_para_la_Celebracion_de_Espectaculos_Publicos_de_la_CDMX_3.4.pdf"),
        ("Ley Orgánica de Alcaldías de la Ciudad de México", "LEY_ORGANICA_DE_ALCALDIAS_DE_LA_CDMX_6.3.pdf"),
        ("Ley de Vivienda para la Ciudad de México", "LEY_DE_VIVIENDA_PARA_LA_CDMX_4.3.pdf"),
        ("Ley de Seguridad Privada para la Ciudad de México", "LEY_DE_SEGURIDAD_PRIVADA_PARA_EL_DF_2.2.pdf"),
        ("Ley de Salud de la Ciudad de México", "2025/2026/210126/LEY_DE_SALUD_DE_LA_CIUDAD_DE_MEXICO_3.7.pdf"),
        ("Ley de Protección y Bienestar de los Animales de la Ciudad de México", "2025/2026/190126/LEY_DE_PROTECCION_Y_BIENESTAR_DE_LOS_ANIMALES_DE_LA_CDMX_1.1.5.pdf"),
        ("Ley de Mejoramiento Barrial y Comunitario de la Ciudad de México", "LEY_DE_MEJORAMIENTO_BARRIAL_Y_COMUNITARIO_DEL_DF_2.3.pdf"),
        ("Ley de Fomento para la Lectura y el Libro de la Ciudad de México", "LEY_DE_FOMENTO_PARA_LA_LECTURA_Y_EL_LIBRO_DE_LA_CIUDAD_DE_MEXICO_6.6.pdf"),
        ("Ley de Entrega Recepción de los Recursos de la Administración Pública de la Ciudad de México", "LEY_DE_ENTREGA_RECEPCION_DE_LOS_RECURSOS_DE_LA_ADMINISTRACION_PUBLICA_DE_LA_CDMX_2.2.pdf"),
        ("Ley de Ingresos de la Ciudad de México para el Ejercicio Fiscal 2025", "leyes/LEY_DE_INGRESOS_2025.pdf"),
        ("Ley de Memoria de la Ciudad de México", "leyes/LEY_DE_MEMORIA_DE_LA_CDMX.pdf"),
        ("Ley del Derecho al Bienestar e Igualdad Social para la Ciudad de México", "leyes/LEY_DEL_DERECHO_AL_BIENESTAR_.pdf"),
        ("Ley de Cultura Cívica de la Ciudad de México", "2025/2026/190126/LEY_DE_CULTURA_CIVICA_DE_LA_CIUDAD_DE_MEXICO_2.8.pdf"),
        ("Ley de Austeridad, Transparencia en Remuneraciones, Prestaciones y Ejercicio de Recursos de la Ciudad de México", "LEY_DE_AUSTERIDAD_TRANSP_EN_REMUNERACIONES_PREST_Y_EJERCICIO_DE_RECURSOS_CDMX_6.2.pdf"),
        ("Ley del Sistema Anticorrupción de la Ciudad de México", "2025/LEY_DEL_SISTEMA_ANTICORRUPCION_DE_LA_CDMX_4.3.pdf"),
        ("Ley del Sistema de Seguridad Ciudadana de la Ciudad de México", "2025/2026/210126/LEY_DEL_SISTEMA_DE_SEGURIDAD_CIUDADANA_DE_LA_CDMX_8.3.pdf"),
        ("Ley de Responsabilidades Administrativas de la Ciudad de México", "2025/2026/210126/LEY_DE_RESPONSABILIDADES_ADMINISTRATIVAS_DE_LA_CDMX_7.4.pdf"),
        ("Ley de Transparencia, Acceso a la Información Pública y Rendición de Cuentas de la Ciudad de México", "2025/LEY_DE_TRANSPARENCIA_ACCESO_A_LA_INFORMACION_PUBLICA_Y_RENDICION_DE_CUENTAS_DE_LA_CDMX_8.4.pdf"),
        ("Ley de Procedimiento Administrativo de la Ciudad de México", "2025/2026/210126/LEY_DE_PROC_ADMINISTRATIVO_DE_LA_CDMX_6.3.pdf"),
        ("Ley Orgánica del Poder Ejecutivo y de la Administración Pública de la Ciudad de México", "2025/2026/210126/LEY_ORG_DEL_P_EJEC_Y_DE_LA_AP_DE_LA_CDMX_14.3.pdf"),
        ("Ley Orgánica del Tribunal de Justicia Administrativa de la Ciudad de México", "2025/LEY_ORGANICA_DEL_TRIBUNAL_DE_JUSTICIA_ADMIN_CDMX_4.6.pdf"),
        ("Ley Orgánica del Poder Judicial de la Ciudad de México", "2025/LEY_ORGANICA_DEL_PODER_JUDICIAL_DE_LA_CDMX_5.5.pdf"),
        ("Ley de Justicia Administrativa de la Ciudad de México", "2025/LEY_DE_JUSTICIA_ADMINISTRATIVA_DE_LA_CDMX_4.4.pdf"),
        ("Ley del Notariado de la Ciudad de México", "2025/2026/50226/LEY_DEL_NOTARIADO_DE_LA_CDMX_5.5.pdf"),
        ("Ley de Planeación del Desarrollo de la Ciudad de México", "LEY_DE_PLANEACION_DEL_DESARROLLO_DE_LA_CDMX_3.3.pdf"),
        ("Ley del Tribunal Superior de Justicia de la Ciudad de México", "leyes/LEY_DEL_TSJ_DE_LA_CDMX_3.1.pdf"),
        ("Ley de Participación Ciudadana de la Ciudad de México", "2025/2026/210126/LEY_DE_PARTICIPACION_CIUDADANA_DE_LA_CDMX_5.2.pdf"),
        ("Ley de Establecimientos Mercantiles de la Ciudad de México", "2025/2026/LEY_DE_ESTABLECIMIENTOS_MERCANTILES_DE_LA_CDMX_7.5.pdf"),
        ("Ley de Desarrollo Urbano del Distrito Federal", "2025/2026/210126/LEY_DE_DESARROLLO_URBANO_DEL_DF_10.5.pdf"),
        ("Ley de Presupuesto y Gasto Eficiente de la Ciudad de México", "2025/LEY_DE_PRESUPUESTO_Y_GASTO_EFICIENTE_DE_LA_CDMX_7.4.pdf"),
        ("Ley del Régimen Patrimonial y del Servicio Público", "LEY_DEL_REGIMEN_PATRIMONIAL_Y_DEL_SERVICIO_PUBLICO_8.3.pdf"),
        ("Ley de Obras Públicas de la Ciudad de México", "2025/LEY_DE_OBRAS_PUBLICAS_DE_LA_CDMX_3.3.pdf"),
        ("Ley de Adquisiciones para el Distrito Federal", "2025/LEY_DE_ADQUISICIONES_PARA_EL_DF_4.3.pdf"),
        ("Ley de Archivos de la Ciudad de México", "2025/LEY_DE_ARCHIVOS_DE_LA_CDMX_3.2.pdf"),
        ("Ley de Derechos de los Pueblos y Barrios Originarios y Comunidades Indígenas Residentes en la Ciudad de México", "2025/181225/LEY_DE_DERECHOS_DE_LOS_PUEBLOS_Y_BARRIOS_ORIGINARIOS_DE_LA_CDMX_3.4.pdf"),
        ("Ley de Propiedad en Condominio de Inmuebles para el Distrito Federal", "2025/2026/210126/LEY_DE_PROPIEDAD_EN_CONDOMINIO_DE_INMUEBLES_PARA_EL_DF_6.7.pdf"),
        ("Ley del Instituto de Verificación Administrativa de la Ciudad de México", "LEY_DEL_INST_DE_VERIFICACION_ADMON_DE_LA_CDMX_5.2.pdf"),
        ("Ley de Publicidad Exterior de la Ciudad de México", "LEY_DE_PUBLICIDAD_EXTERIOR_DE_LA_CDMX_5.4.pdf"),
        ("Ley para la Reconstrucción Integral de la Ciudad de México", "LEY_PARA_LA_RECONSTRUCCION_INTEGRAL_DE_LA_CDMX_4.1.pdf"),
        ("Ley de Residuos Sólidos de la Ciudad de México", "2025/2026/210126/LEY_DE_RESIDUOS_SOLIDOS_DE_LA_CDMX_4.4.pdf"),
        ("Ley de Movilidad de la Ciudad de México", "2025/2026/210126/LEY_DE_MOVILIDAD_DE_LA_CDMX_6.8.pdf"),
        ("Ley para Prevenir y Eliminar la Discriminación de la Ciudad de México", "2025/LEY_PARA_PREVENIR_Y_ELIMINAR_LA_DISCRIMINACION_DE_LA_CDMX_7.3.pdf"),
        ("Ley de Igualdad Sustantiva entre Mujeres y Hombres en la Ciudad de México", "2025/181225/LEY_DE_IGUALDAD_SUSTANTIVA_ENTRE_MUJERES_Y_HOMBRES_EN_LA_CDMX_3.4.pdf"),
        ("Ley de Atención Prioritaria para las Personas con Discapacidad y en Situación de Vulnerabilidad en la Ciudad de México", "LEY_DE_ATENCION_PRIORITARIA_PARA_LAS_PERSONAS_CON_DISCAPACIDAD_DE_LA_CDMX_2.1.pdf"),
        ("Ley de los Derechos de las Personas Jóvenes en la Ciudad de México", "LEY_DE_LOS_DERECHOS_DE_LAS_PERSONAS_JOVENES_EN_LA_CDMX_5.3.pdf"),
        ("Ley de Atención Integral para el Desarrollo de las Niñas y los Niños en Primera Infancia en la Ciudad de México", "leyes/LEY_DE_ATENCION_INTEGRAL_PARA_EL_DESARROLLO_DE_LAS_NINAS_Y_LOS_NINOS_EN_PRIMERA_INFANCIA_EN_LA_CDMX_3.1.pdf"),
        ("Ley de los Derechos de las Personas Adultas Mayores de la Ciudad de México", "2025/LEY_DE_LOS_DERECHOS_DE_LAS_PERSONAS_ADULTAS_MAYORES_DE_LA_CDMX_4.3.pdf"),
        ("Ley para la Integración al Desarrollo de las Personas con Discapacidad de la Ciudad de México", "LEY_PARA_LA_INTEGRACION_AL_DESARROLLO_DE_LAS_PERSONAS_CON_DISCAPACIDAD_DEL_DF_5.2.pdf"),
        ("Ley de Interculturalidad, Atención a Migrantes y Movilidad Humana de la Ciudad de México", "LEY_DE_INTERCULTURALIDAD_ATENCION_A_MIGRANTES_Y_MOV_HUM_EN_LA_CDMX_4.2.pdf"),
        ("Ley de Asistencia y Prevención de la Violencia Familiar", "LEY_DE_ASISTENCIA_Y_PREVENCION_DE_LA_VIOLENCIA_FAMILIAR_3.3.pdf"),
        ("Ley de Responsabilidad Patrimonial del Distrito Federal", "LEY_DE_RESPONSABILIDAD_PATRIMONIAL_DEL_DF_2.2.pdf"),
        ("Ley de Extinción de Dominio para la Ciudad de México", "LEY_DE_EXTINCION_DE_DOMINIO_PARA_LA_CDMX_3.1.pdf"),
        ("Ley de Justicia Alternativa del Tribunal Superior de Justicia para la Ciudad de México", "LEY_DE_JUSTICIA_ALTERNATIVA_DEL_TSJ_PARA_LA_CDMX_6.3.pdf"),
        ("Ley de Mediación, Conciliación y Promoción de la Paz Social para la Ciudad de México", "LEY_DE_MEDIACION_CONCILIACION_Y_PROM_DE_LA_PAZ_SOCIAL_PARA_LA_CDMX_2.3.pdf"),
        ("Ley de Ejecución de Sanciones Penales y Reinserción Social para la Ciudad de México", "2025/LEY_DE_EJEC_DE_SANCIONES_PENALES_Y_REINSERCION_SOCIAL_PARA_LA_CDMX_6.3.pdf"),
        ("Ley de Víctimas para la Ciudad de México", "LEY_DE_VICTIMAS_PARA_LA_CDMX_5.3.pdf"),
        ("Ley de Atención a Personas en Riesgo de Vivir en la Calle y en la Calle de la CDMX", "LEY_DE_ATENCION_A_PERSONAS_EN_RIESGO_DE_VIVIR_EN_LA_CALLE_Y_EN_LA_CALLE_DE_LA_CDMX_2.1.pdf"),
        ("Ley de Cementerios para la Ciudad de México", "LEY_DE_CEMENTERIOS_PARA_LA_CDMX_2.1.pdf"),
        ("Ley de Turismo de la Ciudad de México", "LEY_DE_TURISMO_DE_LA_CDMX_4.3.pdf"),
        ("Ley de Fomento Cultural de la Ciudad de México", "LEY_DE_FOMENTO_CULTURAL_DE_LA_CDMX_4.2.pdf"),
        ("Ley de Filmaciones de la Ciudad de México", "LEY_DE_FILMACIONES_DE_LA_CDMX_2.2.pdf"),
        ("Ley de Economía Social y Solidaria para la Ciudad de México", "LEY_DE_EC_SOCIAL_Y_SOL_PARA_LA_CDMX_2.1.pdf"),
        ("Ley de Fomento Cooperativo para la Ciudad de México", "LEY_DE_FOMENTO_COOPERATIVO_PARA_LA_CDMX_3.2.pdf"),
        ("Ley de Desarrollo Económico de la Ciudad de México", "2025/LEY_DE_DESARROLLO_ECONOMICO_DE_LA_CDMX_3.2.pdf"),
        ("Ley para el Desarrollo del Distrito Federal como Ciudad Digital y del Conocimiento", "LEY_PARA_EL_DESARROLLO_DEL_DF_COMO_CIUDAD_DIGITAL_Y_DEL_CONOCIMIENTO_2.3.pdf"),
        ("Ley de Protección a la Salud de los No Fumadores en el Distrito Federal", "LEY_DE_PROTECCION_A_LA_SALUD_DE_LOS_NO_FUMADORES_EN_EL_DF_2.pdf"),
        ("Ley de Voluntad Anticipada para la Ciudad de México", "LEY_DE_VOLUNTAD_ANTICIPADA_PARA_LA_CDMX_4.2.pdf"),
        ("Ley de Cuidados Alternativos para Niñas, Niños y Adolescentes en la Ciudad de México", "LEY_DE_CUIDADOS_ALTERNATIVOS_PARA_NNA_EN_LA_CDMX_3.1.pdf"),
        ("Ley de Gestión Integral de Riesgos y Protección Civil de la Ciudad de México", "2025/2026/210126/LEY_DE_GESTION_INTEGRAL_DE_RIESGOS_Y_PROTECCION_CIVIL_DE_LA_CDMX_5.5.pdf"),
        ("Ley de Mitigación y Adaptación al Cambio Climático y Desarrollo Sustentable para la Ciudad de México", "LEY_DE_MIT_Y_ADAP_AL_CAMBIO_CLIMATICO_Y_DESARROLLO_SUSTENTABLE_PARA_LA_CDMX_5.3.pdf"),
        ("Ley de Aguas de la Ciudad de México", "LEY_DE_AGUAS_DE_LA_CDMX_3.4.pdf"),
        ("Ley del Derecho al Acceso, Disposición y Saneamiento del Agua de la Ciudad de México", "LEY_DEL_DERECHO_AL_ACCESO_DISPOSICION_Y_SANEAMIENTO_DEL_AGUA_DE_LA_CDMX_2.2.pdf"),
        ("Ley del Sistema de Alerta Sísmica de la Ciudad de México", "leyes/LEY_DEL_SISTEMA_DE_ALERTA_SISMICA_CDMX_1.3.pdf"),
        ("Ley para la Atención Integral de Sustancias Psicoactivas de la CDMX", "LEY_PARA_LA_ATENCION_INTEGRAL_DE_SUSTANCIAS_PSICOACTIVAS_CDMX_2.1.pdf"),
        ("Ley de Impacto Ambiental y Riesgos de la Ciudad de México", "LEY_DE_IMPACTO_AMBIENTAL_Y_RIESGOS_DE_LA_CDMX_2.pdf"),
        ("Ley de Huertos Urbanos en la Ciudad de México", "LEY_DE_HUERTOS_URBANOS_EN_LA_CDMX_2.2.pdf"),
        ("Ley de Economía Circular de la Ciudad de México", "leyes/LEY_DE_ECONOMIA_CIRCULAR_DE_LA_CDMX_1.2.pdf"),
        ("Ley del Sistema de Seguridad Alimentaria y Nutricional de la Ciudad de México", "leyes/LEY_DEL_SIS_DE_SEG_ALIM_Y_NUT_DE_LA_CDMX_2.pdf"),
        ("Ley de Cultura Física y Deporte de la Ciudad de México", "LEY_DE_CULTURA_FISICA_Y_DEPORTE_DE_LA_CDMX_3.4.pdf"),
        ("Ley Orgánica de la Fiscalía General de Justicia de la Ciudad de México", "2025/181225/LEY_ORGANICA_DE_LA_FISCALIA_GENERAL_DE_JUSTICIA_DE_LA_CDMX_7.3.pdf"),
        ("Ley del Heroico Cuerpo de Bomberos de la Ciudad de México", "LEY_DEL_HEROICO_CUERPO_DE_BOMBEROS_DE_LA_CDMX_3.1.pdf"),
        ("Ley de la Comisión de Derechos Humanos de la Ciudad de México", "2025/LEY_DE_LA_COMISION_DE_DERECHOS_HUMANOS_DE_LA_CDMX_5.5.pdf"),
        ("Ley del Servicio Público de Carrera de la Administración Pública de la Ciudad de México", "LEY_DEL_SERVICIO_PUBLICO_DE_CARRERA_DE_LA_AP_DE_LA_CDMX_3.2.pdf"),
        ("Ley de Infraestructura Física Educativa de la Ciudad de México", "LEY_DE_INFRAESTRUCTURA_FISICA_EDUCATIVA_DE_LA_CDMX_3.2.pdf"),
        ("Ley del Centro de Conciliación Laboral de la Ciudad de México", "leyes/LEY_DEL_CENTRO_DE_CONCILIACION_LABORAL_DE_LA_CDMX_2.1.pdf"),
        ("Ley de Espacios Culturales Independientes de la Ciudad de México", "leyes/LEY_DE_ESP_CULTURALES_INDEPENDIENTES_DE_LA_CDMX_2.2.pdf"),
        ("Ley de Ciencia, Tecnología e Innovación de la Ciudad de México", "LEY_DE_CIENCIA_TECNOLOGIA_E_INNOVACION_DE_LA_CDMX_3.2.pdf"),
        ("Ley Constitucional de Derechos Humanos y sus Garantías de la Ciudad de México", "LEY_CONSTITUCIONAL_DE_DERECHOS_HUMANOS_Y_SUS_GARANTIAS_DE_LA_CDMX_3.2.pdf"),
        ("Ley de Gobierno Electrónico de la Ciudad de México", "leyes/LEY_DE_GOBIERNO_ELECTRONICO_DE_LA_CDMX_3.2.pdf"),
        ("Ley de Operación e Innovación Digital de la Ciudad de México", "LEY_DE_OPERACION_E_INNOVACION_DIGITAL_DE_LA_CDMX_2.2.pdf"),
        ("Ley Orgánica de la Administración Pública de la Ciudad de México", "LEY_ORG_DE_LA_ADMON_PUBLICA_DE_LA_CDMX_1.pdf"),
        ("Ley del Instituto de Estudios Constitucionales de la Ciudad de México", "leyes/LEY_DEL_INST_DE_ESTUDIOS_CONSTITUCIONALES_DE_LA_CDMX.pdf"),
        ("Ley Orgánica del Centro de Conciliación Laboral de la Ciudad de México", "leyes/LEY_ORG_DEL_CENTRO_DE_CONCILIACION_LABORAL_DE_LA_CDMX_2.pdf"),
    ]
    for nombre, path in ley_entries:
        LAWS.append(LawDef(
            nombre=nombre,
            url=f"{CDMX_BASE}/{path}",
            categoria="ley",
        ))

    # Códigos (8)
    cod_entries = [
        ("Código Fiscal de la Ciudad de México", "2025/2026/210126/CODIGO_FISCAL_DE_LA_CDMX_26.1.pdf"),
        ("Código de Instituciones y Procedimientos Electorales de la Ciudad de México", "codigos/CODIGO_DE_INSTITUCIONES_Y_PROC_ELECTORALES_CDMX_6.pdf"),
        ("Código de Ética de la Administración Pública de la Ciudad de México", "codigos/CODIGO_DE_ETICA_DE_LA_AP_DE_LA_CIUDAD_DE_MEXICO_2.1.pdf"),
        ("Código de Responsabilidad Parlamentaria del Congreso de la Ciudad de México", "codigos/CODIGO_DE_RESPONSABILIDAD_PARLAMENTARIA_DEL_CONGRESO_DE_LA_CIUDAD_DE_MEXICO_2.1.pdf"),
        ("Código de Procedimientos Civiles para el Distrito Federal", "codigos/Codigo_Procedimientos_Civiles_DF_2.2.pdf"),
        ("Código Civil para el Distrito Federal", "2025/CODIGOS_2025/CODIGO_CIVIL_PARA_EL_DF_15.3.pdf"),
        ("Código Penal para el Distrito Federal", "2025/2026/50226/CODIGO_PENAL_PARA_EL_DF_12.3.4.pdf"),
        ("Código de Ética de los Servidores Públicos para el Distrito Federal", "codigos/Codigo_de_etica_de_los_servidores_publicos_del_DF_3.pdf"),
    ]
    for nombre, path in cod_entries:
        LAWS.append(LawDef(
            nombre=nombre,
            url=f"{CDMX_BASE}/{path}",
            categoria="codigo",
        ))

    # Reglamentos (25)
    reg_entries = [
        ("Reglamento de la Ley de Turismo de la Ciudad de México", "reglamentos/RGTO_DE_LA_LEY_DE_TURISMO_DE_LA_CDMX_1.pdf"),
        ("Reglamento del Registro Civil de la Ciudad de México", "reglamentos/REGLAMENTO_DEL_REGISTRO_CIVIL_DE_LA_CDMX_1.1.pdf"),
        ("Reglamento de la Ley de Establecimientos Mercantiles para la Ciudad de México", "reglamentos/REGLAMENTO_DE_LA_LEY_DE_ESTABLECIMIENTOS_MERCANTILES_PARA_LA_CIUDAD_DE_MEXICO_2.pdf"),
        ("Reglamento de la Ley de Cultura Cívica de la Ciudad de México", "reglamentos/REGLAMENTO_DE_LA_LEY_DE_CULTURA_CIVICA_DE_LA_CDMX.pdf"),
        ("Reglamento de la Ley del Notariado en la Ciudad de México", "reglamentos/REGLAMENTO_DE_LA_LEY_DEL_NOTARIADO_EN_LA_CIUDAD_DE_MEXICO_1.pdf"),
        ("Reglamento de la Ley de Publicidad Exterior de la Ciudad de México", "reglamentos/RTO_LEY_DE_PUBLICIDAD_EXTERIOR_CDMX_2.6.pdf"),
        ("Reglamento de la Ley de Huertos Urbanos de la Ciudad de México", "reglamentos/rgto_de_la_ley_de_huertos_urbanos_cdmx.pdf"),
        ("Reglamento de la Ley de Economía Circular de la Ciudad de México", "reglamentos/RTO_ECONOMIA_CIRCULAR_CDMX.pdf"),
        ("Reglamento de la Ley de Espacios Culturales Independientes de la Ciudad de México", "reglamentos/RTO_LEY_ESP_CULTURALES_IND_CDMX.2.pdf"),
        ("Reglamento de Construcciones para la Ciudad de México", "reglamentos/REGLAMENTO_DE_CONSTRUCCIONES_PARA_LA_CDMX_4.4.pdf"),
        ("Reglamento de la Ley Ambiental del Distrito Federal", "reglamentos/RGTO_DE_LA_LEY_AMBIENTAL_DEL_DF_2.2.pdf"),
        ("Reglamento de la Ley de Residuos Sólidos para el Distrito Federal", "reglamentos/RGTO_LEY_DE_RESIDUOS_SOLIDOS_2.1.pdf"),
        ("Reglamento de Tránsito de la Ciudad de México", "reglamentos/REGLAMENTO_DE_TRANSITO_DE_LA_CDMX_4.3.pdf"),
        ("Reglamento de la Ley de Movilidad de la Ciudad de México", "reglamentos/REGLAMENTO_DE_LA_LEY_DE_MOVILIDAD_DE_LA_CDMX.pdf"),
        ("Reglamento de la Ley de Salud de la Ciudad de México", "reglamentos/RGTO_DE_LA_LEY_DE_SALUD_DEL_DF_3.pdf"),
        ("Reglamento de la Ley de Protección de Datos Personales en Posesión de Sujetos Obligados de la CDMX", "reglamentos/RGTO_LEY_PROTECCION_DATOS_PERSONALES_CDMX.pdf"),
        ("Reglamento de la Ley de Desarrollo Urbano del Distrito Federal", "reglamentos/RGTO_DE_LA_LEY_DE_DESARROLLO_URBANO_DEL_DF_2.3.pdf"),
        ("Reglamento de Cementerios del Distrito Federal", "reglamentos/RGTO_DE_CEMENTERIOS_DEL_DF_1.1.pdf"),
        ("Reglamento de la Ley de Obras Públicas de la Ciudad de México", "reglamentos/RGTO_DE_LA_LEY_DE_OBRAS_PUBLICAS_DE_LA_CDMX_2.pdf"),
        ("Reglamento de la Ley de Adquisiciones para el Distrito Federal", "reglamentos/RGTO_LEY_DE_ADQUISICIONES_PARA_EL_DF_2.1.pdf"),
        ("Reglamento de la Ley de Protección y Bienestar de los Animales de la CDMX", "reglamentos/RGTO_DE_LA_LEY_DE_PROTECCION_A_LOS_ANIMALES_DEL_DF_1.1.pdf"),
        ("Reglamento de la Ley de Gestión Integral de Riesgos y Protección Civil de la CDMX", "reglamentos/RGTO_DE_LA_LEY_DE_GESTION_INTEGRAL_DE_RIESGOS_Y_PROTECCION_CIVIL_1.pdf"),
        ("Reglamento de la Ley de Archivos de la Ciudad de México", "reglamentos/REGLAMENTO_DE_LA_LEY_DE_ARCHIVOS_DE_LA_CDMX.pdf"),
        ("Reglamento de la Ley de Vivienda para la Ciudad de México", "reglamentos/RGTO_DE_LA_LEY_DE_VIVIENDA_DE_LA_CDMX.pdf"),
        ("Reglamento de la Ley de Propiedad en Condominio de Inmuebles del Distrito Federal", "reglamentos/RGTO_DE_LA_LEY_DE_PROPIEDAD_EN_CONDOMINIO_DE_INMUEBLES_DEL_DF_1.2.pdf"),
    ]
    for nombre, path in reg_entries:
        LAWS.append(LawDef(
            nombre=nombre,
            url=f"{CDMX_BASE}/{path}",
            categoria="reglamento",
        ))

    # Assign tipo_codigo
    for law in LAWS:
        law.tipo_codigo = _infer_tipo_codigo(law.nombre, law.categoria)

_build_laws()

# ══════════════════════════════════════════════════════════════════════
# ARTICLE-AWARE CHUNKING
# ══════════════════════════════════════════════════════════════════════

# Regex to detect article boundaries in Mexican legal text
ARTICLE_PATTERN = re.compile(
    r'(?:^|\n)'                           # Start of line
    r'(Art[ií]culo\s+\d+[\w]*'            # "Artículo 15", "Artículo 15 Bis"
    r'(?:\s+(?:BIS|TER|QUÁTER|QUINQUIES))?' # Optional suffixes
    r'[\.\-\s])',                          # Followed by period, dash or space
    re.IGNORECASE | re.MULTILINE
)

# Regex to extract article reference
ARTICLE_REF_PATTERN = re.compile(
    r'Art[ií]culo\s+(\d+[\w]*(?:\s+(?:BIS|TER|QUÁTER|QUINQUIES))?)',
    re.IGNORECASE
)

# Regex to detect section headers (títulos, capítulos, secciones)
SECTION_PATTERN = re.compile(
    r'(?:^|\n)\s*((?:TÍTULO|CAPITULO|CAPÍTULO|SECCIÓN|SECCION|LIBRO)\s+[IVXLCDM\d]+)',
    re.IGNORECASE | re.MULTILINE
)

@dataclass
class Chunk:
    """A single chunk ready for embedding."""
    text: str
    origin: str            # Law name
    ref: str               # Article reference (e.g., "Art. 15")
    jerarquia_txt: str     # Hierarchical context
    tipo_codigo: str       # Code type for Da Vinci
    jurisdiccion: str      # Legal matter
    categoria: str         # constitucion, ley, codigo, ley_organica
    chunk_index: int = 0   # Sub-chunk index (for long articles)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = pymupdf.open(str(pdf_path))
        pages = []
        for page in doc:
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        doc.close()
        return "\n".join(pages)
    except Exception as e:
        print(f"   ❌ Error reading PDF {pdf_path.name}: {e}")
        return ""


def article_aware_chunk(text: str, law: LawDef) -> list[Chunk]:
    """
    Split legal text into article-aware chunks.
    
    Strategy:
    1. Split by article boundaries
    2. Each article = 1 chunk (if < MAX_CHUNK_TOKENS)
    3. Long articles → sub-chunks with overlap
    4. Non-article text (preambles, transitorios) → fixed-size chunks
    """
    if not text.strip():
        return []
    
    # Clean the text
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)  # Collapse horizontal whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
    
    chunks: list[Chunk] = []
    jurisdiccion = _infer_jurisdiccion(law.nombre)
    
    # Track the current section header
    current_section = ""
    
    # Split by article boundaries
    splits = ARTICLE_PATTERN.split(text)
    
    if len(splits) <= 1:
        # No articles found — chunk as fixed-size
        return _fixed_size_chunk(text, law, jurisdiccion)
    
    # Process preamble (text before first article)
    preamble = splits[0].strip()
    if preamble and len(preamble) > MIN_CHUNK_LEN:
        # Extract any section headers from preamble
        sec_match = SECTION_PATTERN.search(preamble)
        if sec_match:
            current_section = sec_match.group(1).strip()
        
        for i, sub in enumerate(_split_long_text(preamble)):
            chunks.append(Chunk(
                text=sub,
                origin=law.nombre,
                ref="Preámbulo",
                jerarquia_txt=f"{law.nombre} > Preámbulo",
                tipo_codigo=law.tipo_codigo,
                jurisdiccion=jurisdiccion,
                categoria=law.categoria,
                chunk_index=i,
            ))
    
    # Process article-text pairs
    # splits alternates: [preamble, "Artículo X.", article_text, "Artículo Y.", article_text, ...]
    i = 1
    while i < len(splits):
        art_header = splits[i].strip() if i < len(splits) else ""
        art_body = splits[i + 1].strip() if (i + 1) < len(splits) else ""
        
        # Full article text
        full_article = f"{art_header} {art_body}".strip()
        
        if len(full_article) < MIN_CHUNK_LEN:
            i += 2
            continue
        
        # Extract article reference
        ref_match = ARTICLE_REF_PATTERN.search(art_header)
        art_ref = f"Art. {ref_match.group(1)}" if ref_match else art_header[:30]
        
        # Update section if we find a header
        sec_match = SECTION_PATTERN.search(art_body)
        if sec_match:
            current_section = sec_match.group(1).strip()
        
        jerarquia = f"{law.nombre} > {current_section} > {art_ref}" if current_section else f"{law.nombre} > {art_ref}"
        
        # Split long articles into sub-chunks
        sub_texts = _split_long_text(full_article)
        for j, sub in enumerate(sub_texts):
            chunks.append(Chunk(
                text=sub,
                origin=law.nombre,
                ref=art_ref,
                jerarquia_txt=jerarquia,
                tipo_codigo=law.tipo_codigo,
                jurisdiccion=jurisdiccion,
                categoria=law.categoria,
                chunk_index=j,
            ))
        
        i += 2
    
    # Check for transitorios after the last article
    if splits and len(splits) > 2:
        last_text = splits[-1]
        trans_match = re.search(r'(?:^|\n)(TRANSITORIOS?)\s*\n', last_text, re.IGNORECASE)
        if trans_match:
            trans_text = last_text[trans_match.start():].strip()
            if len(trans_text) > MIN_CHUNK_LEN:
                for j, sub in enumerate(_split_long_text(trans_text)):
                    chunks.append(Chunk(
                        text=sub,
                        origin=law.nombre,
                        ref="Transitorios",
                        jerarquia_txt=f"{law.nombre} > Transitorios",
                        tipo_codigo=law.tipo_codigo,
                        jurisdiccion=jurisdiccion,
                        categoria=law.categoria,
                        chunk_index=j,
                    ))
    
    return chunks


def _split_long_text(text: str, max_chars: int = 4800) -> list[str]:
    """Split text that exceeds max_chars into overlapping chunks."""
    if len(text) <= max_chars:
        return [text]
    
    parts = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            parts.append(text[start:])
            break
        
        # Try to split at a paragraph or sentence boundary
        split_point = text.rfind('\n\n', start + max_chars // 2, end)
        if split_point == -1:
            split_point = text.rfind('. ', start + max_chars // 2, end)
        if split_point == -1:
            split_point = text.rfind(' ', start + max_chars // 2, end)
        if split_point == -1:
            split_point = end
        else:
            split_point += 1  # Include the delimiter
        
        parts.append(text[start:split_point])
        start = split_point - OVERLAP_CHARS  # Overlap
    
    return parts


def _fixed_size_chunk(text: str, law: LawDef, jurisdiccion: str) -> list[Chunk]:
    """Fallback: chunk text into fixed-size pieces when no articles are detected."""
    chunks = []
    parts = _split_long_text(text, max_chars=3200)
    for i, part in enumerate(parts):
        if len(part.strip()) < MIN_CHUNK_LEN:
            continue
        chunks.append(Chunk(
            text=part.strip(),
            origin=law.nombre,
            ref=f"Sección {i + 1}",
            jerarquia_txt=f"{law.nombre} > Sección {i + 1}",
            tipo_codigo=law.tipo_codigo,
            jurisdiccion=jurisdiccion,
            categoria=law.categoria,
            chunk_index=0,
        ))
    return chunks


# ══════════════════════════════════════════════════════════════════════
# EMBEDDING + QDRANT
# ══════════════════════════════════════════════════════════════════════

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)


async def get_dense_embeddings(texts: list[str]) -> list[list[float]]:
    """Get dense embeddings from OpenAI in batch."""
    # Truncate texts that are too long for the API (max ~8191 tokens)
    truncated = [t[:30000] for t in texts]  # ~7500 tokens
    
    resp = await openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=truncated,
    )
    return [d.embedding for d in resp.data]


async def embed_all_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Embed all chunks with rate limiting and batching."""
    all_embeddings: list[list[float]] = [[] for _ in chunks]
    
    # Process in batches
    for batch_start in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch_end = min(batch_start + EMBED_BATCH_SIZE, len(chunks))
        batch_texts = [c.text for c in chunks[batch_start:batch_end]]
        
        try:
            batch_embeddings = await get_dense_embeddings(batch_texts)
            for i, emb in enumerate(batch_embeddings):
                all_embeddings[batch_start + i] = emb
            
            progress = min(batch_end, len(chunks))
            print(f"   📊 Embedded {progress}/{len(chunks)} chunks")
            
        except Exception as e:
            print(f"   ❌ Embedding error at batch {batch_start}: {e}")
            # Retry with smaller batches
            for j in range(batch_start, batch_end):
                try:
                    embs = await get_dense_embeddings([chunks[j].text])
                    all_embeddings[j] = embs[0]
                except Exception as e2:
                    print(f"   ❌ Skip chunk {j}: {e2}")
                    all_embeddings[j] = [0.0] * EMBEDDING_DIM
        
        # Rate limit: 500 RPM for text-embedding-3-small
        await asyncio.sleep(0.15)
    
    return all_embeddings


def generate_point_id(law_name: str, ref: str, chunk_index: int) -> str:
    """Generate a deterministic UUID for a chunk."""
    raw = f"{ENTIDAD}::{law_name}::{ref}::{chunk_index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


# ══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

async def download_pdfs() -> dict[str, Path]:
    """Download all PDFs to local directory."""
    PDF_DIR.mkdir(exist_ok=True)
    
    downloaded = {}
    failed = []
    
    async with httpx.AsyncClient(timeout=60, follow_redirects=True, verify=False) as client:
        for i, law in enumerate(LAWS):
            filename = law.url.split("/")[-1]
            filepath = PDF_DIR / filename
            
            # Skip if already downloaded
            if filepath.exists() and filepath.stat().st_size > 100:
                downloaded[law.nombre] = filepath
                continue
            
            try:
                resp = await client.get(law.url)
                if resp.status_code == 200 and len(resp.content) > 100:
                    filepath.write_bytes(resp.content)
                    downloaded[law.nombre] = filepath
                    if (i + 1) % 20 == 0:
                        print(f"   📥 Downloaded {i + 1}/{len(LAWS)} PDFs")
                else:
                    failed.append((law.nombre, f"HTTP {resp.status_code}"))
            except Exception as e:
                failed.append((law.nombre, str(e)))
            
            await asyncio.sleep(0.1)  # Rate limit
    
    print(f"\n   ✅ Downloaded: {len(downloaded)}/{len(LAWS)}")
    if failed:
        print(f"   ❌ Failed: {len(failed)}")
        for name, err in failed[:5]:
            print(f"      • {name}: {err}")
    
    return downloaded


def delete_cdmx_data(qdrant: QdrantClient):
    """Delete all existing Querétaro chunks from leyes_estatales."""
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 1: DELETING EXISTING CDMX DATA")
    print("═══════════════════════════════════════════════════════════════")
    
    try:
        # Count existing
        total = qdrant.count(collection_name=COLLECTION)
        print(f"   📊 Total points in collection: {total.count}")
        
        if total.count == 0:
            print("   ✅ Collection is empty, nothing to delete")
            return
        
        # Try filtered count
        count = qdrant.count(
            collection_name=COLLECTION,
            count_filter=Filter(
                must=[FieldCondition(key="entidad", match=MatchValue(value=ENTIDAD))]
            ),
        )
        print(f"   📊 Existing CDMX chunks: {count.count}")
        
        if count.count == 0:
            print("   ✅ No CDMX data to delete")
            return
        
        # Delete by filter
        qdrant.delete(
            collection_name=COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="entidad", match=MatchValue(value=ENTIDAD))]
            ),
        )
        print(f"   ✅ Deleted {count.count} CDMX chunks")
        
    except Exception as e:
        print(f"   ⚠️ Delete phase skipped (collection may be empty/new): {e}")


async def run_ingestion():
    """Main ingestion pipeline."""
    start_time = time.time()
    
    print("═══════════════════════════════════════════════════════════════")
    print("  CDMX LAW INGESTION — Article-Aware Pipeline")
    print(f"  Laws: {len(LAWS)} | Collection: {COLLECTION}")
    print("═══════════════════════════════════════════════════════════════")
    
    # Connect to Qdrant
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Phase 1: Delete existing data
    delete_cdmx_data(qdrant)
    
    # Phase 2: Download PDFs
    if "--skip-download" not in sys.argv:
        print("\n═══════════════════════════════════════════════════════════════")
        print("  PHASE 2: DOWNLOADING PDFs")
        print("═══════════════════════════════════════════════════════════════")
        downloaded = await download_pdfs()
    else:
        # Use cached PDFs
        downloaded = {}
        for law in LAWS:
            filename = law.url.split("/")[-1]
            filepath = PDF_DIR / filename
            if filepath.exists():
                downloaded[law.nombre] = filepath
        print(f"\n   📁 Using cached PDFs: {len(downloaded)}")
    
    # Phase 3: Extract + Chunk
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 3: EXTRACTING TEXT + ARTICLE-AWARE CHUNKING")
    print("═══════════════════════════════════════════════════════════════")
    
    all_chunks: list[Chunk] = []
    laws_processed = 0
    laws_failed = 0
    
    for law in LAWS:
        if law.nombre not in downloaded:
            print(f"   ⚠️  Skipping {law.nombre} (not downloaded)")
            laws_failed += 1
            continue
        
        filepath = downloaded[law.nombre]
        text = extract_text_from_pdf(filepath)
        
        if not text.strip():
            print(f"   ⚠️  Empty PDF: {law.nombre}")
            laws_failed += 1
            continue
        
        chunks = article_aware_chunk(text, law)
        all_chunks.extend(chunks)
        laws_processed += 1
        
        if laws_processed % 20 == 0:
            print(f"   📄 Processed {laws_processed}/{len(LAWS)} laws, {len(all_chunks)} chunks so far")
    
    print(f"\n   📊 CHUNKING SUMMARY:")
    print(f"      Laws processed: {laws_processed}")
    print(f"      Laws failed: {laws_failed}")
    print(f"      Total chunks: {len(all_chunks)}")
    
    # Stats by category
    cats = {}
    for c in all_chunks:
        cats[c.categoria] = cats.get(c.categoria, 0) + 1
    for cat, count in sorted(cats.items()):
        print(f"      {cat}: {count} chunks")
    
    if not all_chunks:
        print("   ❌ No chunks generated! Aborting.")
        return
    
    # Phase 4: Generate embeddings
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 4: GENERATING DENSE EMBEDDINGS")
    print("═══════════════════════════════════════════════════════════════")
    
    embeddings = await embed_all_chunks(all_chunks)
    
    # Phase 5: Upsert to Qdrant
    print("\n═══════════════════════════════════════════════════════════════")
    print("  PHASE 5: UPSERTING TO QDRANT")
    print("═══════════════════════════════════════════════════════════════")
    
    points = []
    for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
        point_id = generate_point_id(chunk.origin, chunk.ref, chunk.chunk_index)
        
        payload = {
            "entidad": ENTIDAD,
            "origen": chunk.origin,
            "ref": chunk.ref,
            "texto": chunk.text,
            "jerarquia_txt": chunk.jerarquia_txt,
            "tipo_codigo": chunk.tipo_codigo,
            "jurisdiccion": chunk.jurisdiccion,
            "categoria": chunk.categoria,
            "chunk_index": chunk.chunk_index,
        }
        
        # Use named vectors (matching collection schema)
        point = PointStruct(
            id=point_id,
            vector={
                "dense": embedding,
            },
            payload=payload,
        )
        points.append(point)
    
    # Batch upsert
    for batch_start in range(0, len(points), QDRANT_BATCH_SIZE):
        batch = points[batch_start:batch_start + QDRANT_BATCH_SIZE]
        try:
            qdrant.upsert(
                collection_name=COLLECTION,
                points=batch,
            )
            progress = min(batch_start + QDRANT_BATCH_SIZE, len(points))
            print(f"   ✅ Upserted {progress}/{len(points)} points")
        except Exception as e:
            print(f"   ❌ Upsert error at batch {batch_start}: {e}")
            # Try individual upserts
            for p in batch:
                try:
                    qdrant.upsert(collection_name=COLLECTION, points=[p])
                except Exception as e2:
                    print(f"      ❌ Point {p.id}: {e2}")
    
    # Verification
    print("\n═══════════════════════════════════════════════════════════════")
    print("  VERIFICATION")
    print("═══════════════════════════════════════════════════════════════")
    
    final_count = qdrant.count(
        collection_name=COLLECTION,
        count_filter=Filter(
            must=[FieldCondition(key="entidad", match=MatchValue(value=ENTIDAD))]
        ),
    )
    
    elapsed = time.time() - start_time
    
    print(f"\n   ✅ INGESTION COMPLETE")
    print(f"      CDMX chunks in Qdrant: {final_count.count}")
    print(f"      Time elapsed: {elapsed:.1f}s")
    print(f"\n   ⚠️  NEXT STEP: Trigger BM25 sparse vector generation via:")
    print(f"      POST https://api.iurexia.com/admin/reingest-sparse")
    print(f"      Body: {{\"admin_key\": \"...\", \"entidad\": \"CDMX\"}}")


async def main():
    if "--delete-only" in sys.argv:
        qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        delete_cdmx_data(qdrant)
    else:
        await run_ingestion()


if __name__ == "__main__":
    asyncio.run(main())
