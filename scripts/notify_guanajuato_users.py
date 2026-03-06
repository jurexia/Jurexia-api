import os
import yaml
import asyncio
import httpx
from supabase import create_client

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Load credentials from the backend config
config_path = r'c:\Users\jdmju\.gemini\antigravity\playground\obsidian-expanse\jurexia-api-git\.env.cloudrun.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

SUPABASE_URL = config.get('SUPABASE_URL')
SUPABASE_KEY = config.get('SUPABASE_SERVICE_KEY')
RESEND_API_KEY = os.environ.get('RESEND_API_KEY')  # Needs to be provided

ENTIDAD_OBJETIVO = 'GUANAJUATO'
FROM_EMAIL = 'Iurexia <noreply@iurexia.com>'

# ══════════════════════════════════════════════════════════════════════════════
# EMAIL TEMPLATE
# ══════════════════════════════════════════════════════════════════════════════

def build_notification_email(name: str):
    first_name = name.split(' ')[0]
    return f"""
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="margin:0;padding:0;background-color:#0a0a0a;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
    <table width="100%" cellpadding="0" cellspacing="0" style="background-color:#0a0a0a;padding:40px 20px;">
        <tr>
            <td align="center">
                <table width="600" cellpadding="0" cellspacing="0" style="background-color:#111111;border-radius:16px;overflow:hidden;border:1px solid #222;">
                    <tr>
                        <td style="background:linear-gradient(135deg,#1a1a1a 0%,#1f1f1f 100%);padding:32px 40px;border-bottom:1px solid #333;">
                            <span style="font-size:26px;font-weight:800;color:#ffffff;letter-spacing:1px;">
                                IUREX<span style="color:#c9a84c;">IA</span>
                            </span>
                        </td>
                    </tr>
                    <tr>
                        <td style="padding:40px;">
                            <h1 style="margin:0 0 16px;font-size:24px;font-weight:700;color:#ffffff;">
                                ¡Guanajuato ya está disponible, {first_name}! ⚖️
                            </h1>
                            <p style="margin:0 0 24px;font-size:16px;color:#ccc;line-height:1.6;">
                                Tenemos excelentes noticias para tu práctica legal. La legislación del estado de <strong>Guanajuato</strong> ha sido integrada completamente en nuestra base de datos.
                            </p>
                            
                            <div style="background-color:#1a1a1a;border:1px solid #2a2a2a;border-radius:12px;padding:20px;margin-bottom:24px;">
                                <p style="margin:0 0 12px;font-size:14px;font-weight:600;color:#c9a84c;text-transform:uppercase;">
                                    📚 Nuevo contenido disponible:
                                </p>
                                <ul style="margin:0;padding-left:20px;color:#bbb;font-size:14px;line-height:1.8;">
                                    <li>Constitución Política de Guanajuato</li>
                                    <li>Más de 100 Leyes Estatales</li>
                                    <li>Códigos: Penal, Civil, Fiscal, Territorial y más</li>
                                    <li>Reglamentos vigentes</li>
                                </ul>
                            </div>

                            <div style="background-color:#1a1520;border:1px solid #2d2040;border-radius:12px;padding:20px;margin-bottom:28px;">
                                <p style="margin:0 0 8px;font-size:14px;font-weight:700;color:#c4b5fd;">
                                    💡 Tip para resultados precisos
                                </p>
                                <p style="margin:0;font-size:14px;color:#bbb;line-height:1.6;">
                                    Recuerda utilizar el <strong>filtro de Fuero: Local</strong>. Esto le indica a Iurexia que debe priorizar los códigos y leyes de Guanajuato en tus consultas.
                                </p>
                            </div>

                            <table cellpadding="0" cellspacing="0" width="100%">
                                <tr>
                                    <td align="center">
                                        <a href="https://www.iurexia.com/chat"
                                           style="display:inline-block;background:linear-gradient(135deg,#c9a84c,#e8c56d);color:#1a1a1a;font-size:15px;font-weight:700;padding:14px 40px;border-radius:10px;text-decoration:none;">
                                            Comenzar consulta local →
                                        </a>
                                    </td>
                               Subsequent steps if needed...
                            </table>
                        </td>
                    </tr>
                    <tr>
                        <td style="background-color:#0a0a0a;padding:24px;border-top:1px solid #222;text-align:center;">
                            <p style="margin:0;font-size:11px;color:#444;">
                                © 2026 Iurexia Technologies. Inteligencia Artificial para el Derecho Mexicano.
                            </p>
                        </td>
                    </tr>
                </table>
            </td>
        </tr>
    </table>
</body>
</html>
"""

# ══════════════════════════════════════════════════════════════════════════════
# EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

async def main():
    if not RESEND_API_KEY:
        print("❌ Error: RESEND_API_KEY no encontrada en el entorno.")
        return

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # 1. Fetch users
    print(f"🔍 Buscando usuarios de {ENTIDAD_OBJETIVO}...")
    res = supabase.table('user_profiles').select('email, full_name').eq('estado', ENTIDAD_OBJETIVO).execute()
    users = res.data
    
    if not users:
        print("ℹ️ No se encontraron usuarios para esta entidad.")
        return
    
    print(f"📧 Se enviarán {len(users)} correos.")
    
    async with httpx.AsyncClient() as client:
        for user in users:
            email = user['email']
            name = user['full_name']
            
            print(f"   📤 Enviando a: {name} <{email}>...")
            
            payload = {
                "from": FROM_EMAIL,
                "to": email,
                "subject": f"📍 ¡Guanajuato ya está disponible en Iurexia! ⚖️",
                "html": build_notification_email(name)
            }
            
            resp = await client.post(
                "https://api.resend.com/emails",
                json=payload,
                headers={"Authorization": f"Bearer {RESEND_API_KEY}"}
            )
            
            if resp.status_code in [200, 201]:
                print(f"      ✅ Éxito")
            else:
                print(f"      ❌ Error: {resp.text}")
            
            await asyncio.sleep(0.5) # Anti-spam slow down

if __name__ == "__main__":
    asyncio.run(main())
