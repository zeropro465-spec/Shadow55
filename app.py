import os
from groq import Groq
from ddgs import DDGS

client = Groq(api_key="")

# Admin şifrəsini burada təyin edirsən
ADMIN_PASSWORD = "admin123"

def web_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(query, max_results=3)]
        return " ".join(results)
    except Exception as e:
        return f"Axtarış zamanı xəta baş verdi: {str(e)}"

def ask_shadow(user_query: str) -> str:
    search_data = web_search(user_query)
    
    system_prompt = f"""
    Sənin adın Shadow-dur. Sən yüksək səviyyəli analitik, internetə çıxışı olan və dərin düşünmə qabiliyyətinə malik süni intellekt köməkçisən.
    Mürəkkəb tapşırıqları yerinə yetirərkən əvvəlcə addım-addım düşün (Chain of Thought).
    İstifadəçinin sualını cavablandırarkən aşağıdakı internetdən əldə edilmiş real vaxt məlumatlarından istifadə et:
    
    [İnternet Məlumatları]:
    {search_data}
    """

    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ],
        model="openai/gpt-oss-120b",
        temperature=0.5,
    )
    
    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    print("--- SHADOW TƏHLÜKƏSİZLİK SİSTEMİ ---")
    giris_parolu = input("Zəhmət olmasa admin şifrəsini daxil edin: ")
    
    if giris_parolu == ADMIN_PASSWORD:
        print("\nGiriş uğurludur! Xoş gəldin, Admin. Shadow hazırdır (çıxmaq üçün 'q' yaz).")
        while True:
            sual = input("\nAdmin Sualı: ")
            if sual.lower() == 'q':
                print("Shadow söndürüldü.")
                break
            
            cavab = ask_shadow(sual)
            print("\nShadow-nun Cavabı:\n")
            print(cavab)
    else:
        print("\nXəta: Yanlış şifrə! Yalnız admin səlahiyyəti olan şəxslər Shadow-a sual verə bilər.")
