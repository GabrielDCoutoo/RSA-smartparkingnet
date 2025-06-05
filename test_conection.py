#!/usr/bin/env python3
"""
Script para testar conectividade e diagnosticar problemas
"""

import socket
import subprocess
import sys
import time
import os

def test_port(host, port):
    """Testa se uma porta está disponível"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(5)
            result = s.connect_ex((host, port))
            return result == 0
    except Exception as e:
        print(f"Erro ao testar porta {port}: {e}")
        return False

def find_python_processes():
    """Encontra processos Python relacionados com Flower"""
    try:
        output = subprocess.check_output(['ps', 'aux'], universal_newlines=True)
        flower_processes = []
        for line in output.split('\n'):
            if 'python' in line and ('flower' in line or 'serverflower' in line or 'clientML' in line):
                flower_processes.append(line.strip())
        return flower_processes
    except:
        return []

def check_dependencies():
    """Verifica se as dependências estão instaladas"""
    dependencies = ['flwr', 'tensorflow', 'pandas', 'numpy', 'sklearn']
    missing = []
    
    for dep in dependencies:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError:
            missing.append(dep)
            print(f"❌ {dep}")
    
    return missing

def main():
    print("🔍 DIAGNÓSTICO DE CONECTIVIDADE FLOWER")
    print("=" * 50)
    
    # 1. Verificar dependências
    print("\n1️⃣ Verificando dependências Python:")
    missing_deps = check_dependencies()
    if missing_deps:
        print(f"\n❌ Dependências em falta: {', '.join(missing_deps)}")
        print("Execute: pip install " + " ".join(missing_deps))
        return
    
    # 2. Verificar processos existentes
    print("\n2️⃣ Verificando processos Flower existentes:")
    processes = find_python_processes()
    if processes:
        print("⚠️ Processos encontrados:")
        for proc in processes:
            print(f"   {proc}")
        print("\nPara terminar todos os processos:")
        print("   pkill -f 'python.*flower'")
    else:
        print("✅ Nenhum processo Flower ativo")
    
    # 3. Testar portas
    print("\n3️⃣ Testando disponibilidade de portas:")
    ports_to_test = [8080, 8081, 8082, 8083, 8084]
    available_ports = []
    
    for port in ports_to_test:
        if not test_port('localhost', port):
            available_ports.append(port)
            print(f"✅ Porta {port} disponível")
        else:
            print(f"❌ Porta {port} ocupada")
    
    if not available_ports:
        print("⚠️ Todas as portas testadas estão ocupadas!")
    
    # 4. Verificar ficheiros necessários
    print("\n4️⃣ Verificando ficheiros:")
    files_to_check = [
        'server/serverflower.py',
        'cmd/clientML.py',
        'smartparknet_dashboard/backend/forecast_b'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    # 5. Teste básico de servidor
    print("\n5️⃣ Teste básico de conectividade:")
    if available_ports:
        test_port_number = available_ports[0]
        print(f"Testando servidor básico na porta {test_port_number}...")

        
        # Criar servidor de teste simples
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', test_port_number))
                s.listen(1)
                print(f"✅ Servidor de teste criado com sucesso na porta {test_port}")
                s.close()
        except Exception as e:
            print(f"❌ Erro no servidor de teste: {e}")
    
    # 6. Recomendações
    print("\n6️⃣ Recomendações:")
    print("🔄 Para reiniciar limpo:")
    print("   1. pkill -f 'python.*flower'")
    print("   2. rm -f server_port.txt")
    print("   3. ./startfederated.sh")
    
    print("\n🐛 Para debug:")
    print("   1. Execute o servidor manualmente: python3 serverflower.py")
    print("   2. Em outro terminal, execute um cliente: DATASET=path/to/file.csv python3 cmd/clientML.py")
    
    if missing_deps:
        print(f"\n⚠️ IMPORTANTE: Instale as dependências em falta primeiro!")

if __name__ == "__main__":
    main()