"""Cómo resuelve DE VERDAD este circuito, contado sobre su propio acervo.

David: «un pipeline bien implementado puede superar los sesgos que un
secretario puede llegar a tener por más capacitado y experimentado que sea».
Esto es lo que lo hace posible: no es una opinión sobre cómo se resuelve, es el
recuento de cómo se resolvió en 12,272 expedientes del Vigésimo Segundo
Circuito, con 65,282 agravios calificados uno por uno.

QUÉ SE MIDIÓ. Para cada tipo de asunto, con qué frecuencia sale cada sentido y
—lo que importa— cómo se califican los agravios en cada uno. La correlación es
fuerte y útil:

    revisión que CONFIRMA .... infundado 52% · inoperante 33% · fundado 1%
    revisión que REVOCA ...... fundado 51% · esencialmente fundado 22%

Un proyecto que confirma con un agravio fundado está en el 1% del circuito. No
es imposible —a veces se confirma por razones distintas de las del juzgado—
pero es lo bastante raro como para mirarlo dos veces antes de firmar.

DE DÓNDE SALE. `sentencias_holdings` en Qdrant, que ya traía la extracción por
sentencia: cada agravio con su descripción, su calificación y su tipo. NO hubo
que descargar ni releer nada; el trabajo era agregarlo.

LO QUE ENSEÑÓ Y NO SABÍAMOS: el 8% de las calificaciones reales (5,523 de
65,282) no cabe en nuestras cuatro. La que más falta hace es
«esencialmente fundado», que es el 22% de los agravios en las revisiones que
revocan.
"""
# GENERADO desde sentencias_holdings, circuito 22 · 33,036 holdings · 12,272
# expedientes. NO se edita a mano: se regenera con scratchpad/barrer22.py.
TABLA_CIRCUITO = {
    "amparo_directo": {
        "expedientes": 7601,
        "holdings": 13424,
        "sentidos": {
            "niega": {
                "frecuencia": 52.8,
                "calificaciones": {
                    "infundado": 48,
                    "inoperante": 35,
                    "ineficaz": 12,
                    "fundado": 1,
                    "fundado_insuficiente": 1
                }
            },
            "concede": {
                "frecuencia": 26.2,
                "calificaciones": {
                    "fundado": 50,
                    "esencialmente_fundado": 22,
                    "infundado": 14,
                    "inoperante": 7,
                    "ineficaz": 3
                }
            },
            "sobresee": {
                "frecuencia": 8.0,
                "calificaciones": {
                    "inoperante": 36,
                    "infundado": 34,
                    "fundado": 21,
                    "inatendible": 6,
                    "innecesario_examinar": 1
                }
            },
            "parcialmente_concede": {
                "frecuencia": 6.9,
                "calificaciones": {
                    "fundado": 39,
                    "infundado": 33,
                    "inoperante": 10,
                    "ineficaz": 6,
                    "esencialmente_fundado": 5
                }
            },
            "desecha": {
                "frecuencia": 2.8,
                "calificaciones": {
                    "fundado": 38,
                    "inoperante": 37,
                    "infundado": 23,
                    "incompetente": 1
                }
            }
        }
    },
    "amparo_revision": {
        "expedientes": 5092,
        "holdings": 10976,
        "sentidos": {
            "confirma": {
                "frecuencia": 60.0,
                "calificaciones": {
                    "infundado": 53,
                    "inoperante": 33,
                    "ineficaz": 10,
                    "fundado": 1,
                    "fundado_insuficiente": 1
                }
            },
            "revoca": {
                "frecuencia": 10.6,
                "calificaciones": {
                    "fundado": 52,
                    "esencialmente_fundado": 23,
                    "infundado": 15,
                    "inoperante": 5,
                    "parcialmente_fundado": 1
                }
            },
            "sobresee": {
                "frecuencia": 7.4,
                "calificaciones": {
                    "fundado": 33,
                    "infundado": 29,
                    "inoperante": 22,
                    "esencialmente_fundado": 6,
                    "fundado_insuficiente": 3
                }
            },
            "modifica": {
                "frecuencia": 6.8,
                "calificaciones": {
                    "infundado": 33,
                    "fundado": 29,
                    "inoperante": 16,
                    "esencialmente_fundado": 11,
                    "ineficaz": 6
                }
            },
            "concede": {
                "frecuencia": 4.7,
                "calificaciones": {
                    "fundado": 63,
                    "esencialmente_fundado": 18,
                    "infundado": 13,
                    "inoperante": 2,
                    "ineficaz": 1
                }
            },
            "niega": {
                "frecuencia": 3.1,
                "calificaciones": {
                    "infundado": 68,
                    "inoperante": 18,
                    "ineficaz": 9,
                    "fundado": 4,
                    "esencialmente_fundado": 1
                }
            },
            "desecha": {
                "frecuencia": 2.9,
                "calificaciones": {
                    "infundado": 37,
                    "inoperante": 36,
                    "fundado": 18,
                    "improcedente": 2,
                    "desechado": 1
                }
            }
        }
    },
    "queja": {
        "expedientes": 2955,
        "holdings": 5926,
        "sentidos": {
            "confirma": {
                "frecuencia": 32.9,
                "calificaciones": {
                    "infundado": 60,
                    "inoperante": 24,
                    "ineficaz": 11,
                    "inatendible": 2,
                    "fundado": 2
                }
            },
            "infundado": {
                "frecuencia": 17.5,
                "calificaciones": {
                    "infundado": 60,
                    "ineficaz": 19,
                    "inoperante": 15,
                    "inatendible": 3,
                    "fundado": 1
                }
            },
            "revoca": {
                "frecuencia": 16.4,
                "calificaciones": {
                    "fundado": 61,
                    "esencialmente_fundado": 26,
                    "infundado": 4,
                    "inoperante": 4,
                    "sustancialmente_fundado": 1
                }
            },
            "desecha": {
                "frecuencia": 9.6,
                "calificaciones": {
                    "infundado": 61,
                    "inoperante": 27,
                    "fundado": 7,
                    "sin_materia": 2,
                    "inatendible": 1
                }
            },
            "sin_materia": {
                "frecuencia": 7.3,
                "calificaciones": {
                    "sin_materia": 55,
                    "inoperante": 42,
                    "infundado": 1,
                    "fundado": 1,
                    "no_analizado": 1
                }
            },
            "fundado": {
                "frecuencia": 4.4,
                "calificaciones": {
                    "fundado": 66,
                    "esencialmente_fundado": 24,
                    "infundado": 5,
                    "ineficaz": 2,
                    "inoperante": 1
                }
            },
            "modifica": {
                "frecuencia": 4.0,
                "calificaciones": {
                    "fundado": 45,
                    "infundado": 20,
                    "esencialmente_fundado": 17,
                    "ineficaz": 7,
                    "inoperante": 4
                }
            }
        }
    },
    "revision_fiscal": {
        "expedientes": 444,
        "holdings": 1010,
        "sentidos": {
            "confirma": {
                "frecuencia": 48.2,
                "calificaciones": {
                    "inoperante": 57,
                    "infundado": 31,
                    "ineficaz": 9,
                    "fundado": 1,
                    "fundado_insuficiente": 1
                }
            },
            "desecha": {
                "frecuencia": 34.9,
                "calificaciones": {
                    "infundado": 71,
                    "inoperante": 13,
                    "fundado": 11,
                    "inatendible": 4,
                    "ineficaz": 0
                }
            },
            "revoca": {
                "frecuencia": 10.3,
                "calificaciones": {
                    "fundado": 38,
                    "inoperante": 27,
                    "infundado": 16,
                    "esencialmente_fundado": 13,
                    "ineficaz": 2
                }
            },
            "modifica": {
                "frecuencia": 2.6,
                "calificaciones": {
                    "fundado": 34,
                    "inoperante": 29,
                    "infundado": 18,
                    "esencialmente_fundado": 14,
                    "sustancialmente_fundado": 5
                }
            }
        }
    }
}
