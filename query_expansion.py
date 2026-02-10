"""
query_expansion.py

Módulo de Query Expansion para Jurexia - FASE 1
Analiza queries de usuario y determina qué tipos de documentos recuperar

Author: Antigravity AI
Date: 2026-02-09
"""

import json
from typing import Dict, List, Optional
from openai import AsyncOpenAI
import os


class QueryExpander:
    """
    Analizador de queries legales para expansion inteligente
    """
    
    def __init__(self, deepseek_client: AsyncOpenAI):
        self.client = deepseek_client
    
    async def analyze_query(self, query: str) -> Dict:
        """
        Analizar query y determinar estrategia de búsqueda
        
        Args:
            query: Pregunta del usuario
        
        Returns:
            Dict con estrategia de expansión:
            {
                "requiere_marco_constitucional": bool,
                "articulos_cpeum_relevantes": List[str],
                "requiere_jurisprudencia": bool,
                "temas_jurisprudencia": List[str],
                "requiere_vias_procesales": bool,
                "vias_sugeridas": List[str],
                "materia_principal": str,
                "nivel_profundidad": str  # "basico" | "intermedio" | "avanzado"
            }
        """
        
        expansion_prompt = f"""Analiza esta consulta legal y determina qué tipo de documentos debe buscar el sistema RAG.

Query del usuario:
{query}

Devuelve un JSON con esta estructura exacta:
{{
    "requiere_marco_constitucional": true o false,
    "articulos_cpeum_relevantes": ["lista de números de artículos constitucionales, ej: 1, 14, 16"],
    "requiere_jurisprudencia": true o false,
    "temas_jurisprudencia": ["lista de temas para buscar jurisprudencia"],
    "requiere_vias_procesales": true o false,
    "vias_sugeridas": ["amparo_indirecto", "amparo_directo", "juicio_ordinario", "contencioso_administrativo", etc],
    "materia_principal": "constitucional" | "civil" | "penal" | "mercantil" | "laboral" | "administrativo" | "fiscal" | "familiar",
    "nivel_profundidad": "basico" | "intermedio" | "avanzado",
    "palabras_clave_adicionales": ["términos técnicos o conceptos clave a agregar a la búsqueda"]
}}

Criterios para determinar si requiere marco constitucional:
- Consultas sobre derechos fundamentales (libertad, igualdad, propiedad, etc)
- Temas de constitucionalidad, validez de leyes, inconstitucionalidad
- Garantías individuales o derechos humanos
- División de poderes, federalismo
- Amparo o control constitucional

Criterios para vías procesales:
- Solo si la pregunta implica litigio, impugnación o defensa legal
- NO incluir vías para consultas informativas o de definición

IMPORTANTE: Devuelve SOLO el JSON, sin texto adicional."""

        try:
            # Usar deepseek-reasoner para razonamiento legal complejo
            # Costo: ~$0.55/M tokens (vs $0.14 de chat) pero mejor precisión
            response = await self.client.chat.completions.create(
                model="deepseek-reasoner",
                messages=[{"role": "user", "content": expansion_prompt}],
                temperature=0.2,
                max_tokens=800  # Reasoner necesita más tokens para reasoning
            )
            
            content = response.choices[0].message.content.strip()
            
            # DeepSeek Reasoner puede devolver razonamiento en tags <think>
            # Necesit amos extraer solo el JSON
            if "<think>" in content:
                # Extraer contenido después de </think>
                content = content.split("</think>")[-1].strip()
            
            # Limpiar markdown
            if content.startswith("```json"):
                content = content.split("```json")[1].split("```")[0].strip()
            elif content.startswith("```"):
                content = content.split("```")[1].split("```")[0].strip()
            
            # Si aún tiene tags XML, quitar
            if content.startswith("<"):
                # Buscar primer {
                json_start = content.find("{")
                if json_start != -1:
                    content = content[json_start:]
            
            expansion = json.loads(content)
            
            # Validar estructura
            required_keys = [
                "requiere_marco_constitucional",
                "requiere_jurisprudencia",
                "requiere_vias_procesales",
                "materia_principal"
            ]
            
            for key in required_keys:
                if key not in expansion:
                    print(f"⚠️  Falta clave '{key}' en expansión, usando default")
                    expansion[key] = False if "requiere" in key else "general"
            
            return expansion
            
        except json.JSONDecodeError as e:
            print(f"⚠️  Error parsing expansion JSON: {e}")
            print(f"   Content: {content[:200]}...")
            # Return safe defaults
            return self._get_default_expansion()
        except Exception as e:
            print(f"⚠️  Error en query expansion: {e}")
            return self._get_default_expansion()
    
    def _get_default_expansion(self) -> Dict:
        """Expansión por defecto segura"""
        return {
            "requiere_marco_constitucional": False,
            "articulos_cpeum_relevantes": [],
            "requiere_jurisprudencia": False,
            "temas_jurisprudencia": [],
            "requiere_vias_procesales": False,
            "vias_sugeridas": [],
            "materia_principal": "general",
            "nivel_profundidad": "intermedio",
            "palabras_clave_adicionales": []
        }
    
    def build_expanded_query(self, original_query: str, expansion: Dict) -> str:
        """
        Construir query expandida con términos adicionales
        
        Args:
            original_query: Query original del usuario
            expansion: Resultado de analyze_query()
        
        Returns:
            Query expandida con términos clave
        """
        
        expanded_parts = [original_query]
        
        # Agregar palabras clave adicionales
        if expansion.get("palabras_clave_adicionales"):
            expanded_parts.extend(expansion["palabras_clave_adicionales"])
        
        # Agregar temas de jurisprudencia
        if expansion.get("temas_jurisprudencia"):
            expanded_parts.extend(expansion["temas_jurisprudencia"])
        
        return " ".join(expanded_parts)
    
    def get_search_weights(self, expansion: Dict) -> Dict[str, float]:
        """
        Determinar pesos para merge de resultados según expansión
        
        Args:
            expansion: Resultado de analyze_query()
        
        Returns:
            Dict con pesos para cada tipo de búsqueda
        """
        
        weights = {
            "normal": 0.5,  # Búsqueda normal siempre con peso base
            "constitucional": 0.0,
            "jurisprudencia": 0.0,
        }
        
        # Ajustar pesos según requisitos
        if expansion.get("requiere_marco_constitucional"):
            weights["constitucional"] = 0.3
            weights["normal"] = 0.4
        
        if expansion.get("requiere_jurisprudencia"):
            weights["jurisprudencia"] = 0.3 if not expansion.get("requiere_marco_constitucional") else 0.2
            weights["normal"] = 0.4 if weights["constitucional"] > 0 else 0.4
           
        # Normalizar para que sumen 1.0
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights


# Singleton instance
_query_expander_instance: Optional[QueryExpander] = None


def get_query_expander(deepseek_client: AsyncOpenAI) -> QueryExpander:
    """
    Obtener instancia singleton de QueryExpander
    """
    global _query_expander_instance
    
    if _query_expander_instance is None:
        _query_expander_instance = QueryExpander(deepseek_client)
    
    return _query_expander_instance
