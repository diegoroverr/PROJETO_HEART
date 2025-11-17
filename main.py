# main.py
import os
import sys
import time
import subprocess
from threading import Thread
from queue import Queue, Empty

# Dependências: colorama, tqdm
from colorama import init as colorama_init, Fore, Style
from tqdm import tqdm

colorama_init(autoreset=True)

PY = sys.executable  # caminho do Python atualmente usado

# ---------------------------------------
# Helpers
# ---------------------------------------
def limpar_tela():
    os.system('cls' if os.name == 'nt' else 'clear')

def cabecalho():
    limpar_tela()
    # Título atualizado para o novo tema
    print(Fore.RED + "╔" + "═" * 58 + "╗")
    print(Fore.RED + "║" + Style.BRIGHT + "   ❤️  ANÁLISE DE PERFIS DE INSUFICIÊNCIA CARDÍACA   ❤️  " + Style.NORMAL + Fore.RED + "║")
    print(Fore.RED + "╚" + "═" * 58 + "╝")
    print(Fore.CYAN + "   Identifica perfis de pacientes usando K-Means (sem supervisão)\n" + Style.RESET_ALL)

def imprime_bloco(titulo, texto):
    print(Fore.YELLOW + "─" * 60)
    print(Fore.BLUE + Style.BRIGHT + f" {titulo}")
    print(Fore.YELLOW + "─" * 60)
    print(Style.NORMAL + texto + "\n")

def executar_script_com_progresso(cmd_args, descricao="Executando", timeout=None):
    """
    Executa um subprocess e mostra uma barra de progresso animada enquanto o processo estiver rodando.
    cmd_args: lista ex: [PY, 'normalizar.py']
    """
    # Inicia o processo
    proc = subprocess.Popen(cmd_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Exibe a barra de progresso indefinida (spinner-like) até o processo terminar
    with tqdm(total=0, bar_format="{l_bar}{bar} {remaining}", desc=descricao, leave=True) as pbar:
        # Enquanto o processo não terminou, atualiza barra (pbar.update) para animar
        while proc.poll() is None:
            pbar.update(1)
            time.sleep(0.08)
        # Consome qualquer saída final
        stdout, stderr = proc.communicate()
        # Mostrar saída do script (respeitando cores)
        if stdout:
            print(Fore.WHITE + stdout)
        if stderr:
            print(Fore.RED + "=== ERROS/AVISOS ===")
            print(Fore.RED + stderr)
    return proc.returncode

# ---------------------------------------
# Menu / fluxo
# ---------------------------------------
def menu():
    cabecalho()
    imprime_bloco("", (
        ""
        ""
        "Use as opções abaixo para executar cada etapa do pipeline."
    ))

    print("  " + Fore.GREEN + "[1]" + Style.RESET_ALL + " 🥦  Normalizar dados ")
    print("  " + Fore.GREEN + "[2]" + Style.RESET_ALL + " 🤖  Treinar K-Means ")
    print("  " + Fore.GREEN + "[3]" + Style.RESET_ALL + " 🔍  Descrever centroides ")
    print("  " + Fore.GREEN + "[4]" + Style.RESET_ALL + " 👤  Classificar novo paciente ")
    print("  " + Fore.GREEN + "[9]" + Style.RESET_ALL + " 🧰  Instalar dependências")
    print("  " + Fore.RED   + "[0]" + Style.RESET_ALL + " ❌  Sair\n")
    return input("👉 Escolha uma opção: ").strip()

def instalar_dependencias():
    print(Fore.CYAN + "\nInstalando dependências necessárias (colorama, tqdm) e libs científicas...")
    cmds = [
        [PY, "-m", "pip", "install", "--upgrade", "pip"],
        [PY, "-m", "pip", "install", "colorama", "tqdm", "scikit-learn", "pandas", "numpy", "matplotlib"]
    ]
    for cmd in cmds:
        rc = subprocess.call(cmd)
        if rc != 0:
            print(Fore.RED + f"Erro ao executar: {' '.join(cmd)}")
            return False
    print(Fore.GREEN + "Instalação concluída!\n")
    input("Pressione ENTER para voltar ao menu.")
    return True

def main_loop():
    while True:
        escolha = menu()
        if escolha == '1':
            print(Fore.CYAN + "\n🔄 Normalizando dados...")
            rc = executar_script_com_progresso([PY, "normalizar.py"], descricao="Normalizando")
            if rc == 0:
                print(Fore.GREEN + "✅ Normalização concluída.")
            else:
                print(Fore.RED + f"⚠️ Normalizar retornou código {rc}")
            input("\nPressione ENTER para voltar ao menu.")

        elif escolha == '2':
            print(Fore.CYAN + "\n🧩 Calculando clusters (método do cotovelo + KMeans)...")
            rc = executar_script_com_progresso([PY, "clusterizar.py"], descricao="Treinando KMeans")
            if rc == 0:
                print(Fore.GREEN + "✅ Treinamento concluído.")
            else:
                print(Fore.RED + f"⚠️ clusterizar retornou código {rc}")
            input("\nPressione ENTER para voltar ao menu.")

        elif escolha == '3':
            print(Fore.CYAN + "\n📊 Descrevendo centroides...")
            os.system(f'{PY} descrever_centroides.py')
            print(Fore.GREEN + "\n✅ Centroides exibidos.")
            input("\nPressione ENTER para voltar ao menu.")


        elif escolha == '4':
            print(Fore.CYAN + "\n👤 Classificando um novo paciente (veja o dicionário em processar_paciente_desconhecido.py)...")
            rc = executar_script_com_progresso([PY, "processar_paciente_desconhecido.py"], descricao="Classificando")
            if rc == 0:
                print(Fore.GREEN + "✅ Classificação concluída.")
            else:
                print(Fore.RED + f"⚠️ processar_paciente_desconhecido retornou código {rc}")
            input("\nPressione ENTER para voltar ao menu.")

        elif escolha == '9':
            instalar_dependencias()

        elif escolha == '0':
            print(Fore.MAGENTA + "\n👋 Obrigado por usar o Saúde em Foco. Até logo!")
            break

        else:
            print(Fore.YELLOW + "\nOpção inválida — tente novamente.")
            time.sleep(1.2)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print(Fore.MAGENTA + "\n\nPrograma encerrado pelo usuário. Até mais!")
