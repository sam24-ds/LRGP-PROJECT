"""
python_repl.py
Sandbox d'exécution Python avec env conda dédié SRAR-GP.
Auto-installation de librairies scientifiques manquantes.
"""
import subprocess
import tempfile
import os
import re
from pathlib import Path
from dataclasses import dataclass


# ── Configuration ────────────────────────────────────────────────
# Chemin vers le python de l'env conda dédié
PYTHON_EXE = r"C:\Users\Samir\miniforge3\envs\srar-repl\python.exe"

# Librairies scientifiques autorisées à l'auto-installation
ALLOWED_LIBS = {
    "numpy", "scipy", "sympy", "pandas", "matplotlib",
    "fluids", "thermo", "chemicals", "CoolProp", "ht",
    "pyomo", "cantera",  # optionnels
}

# Imports interdits (sécurité)
FORBIDDEN_PATTERNS = [
    r"\bos\.system\b", r"\bsubprocess\b", r"\beval\s*\(",
    r"\bexec\s*\(", r"\b__import__\b", r"\bopen\s*\(",
    r"\bshutil\b", r"\bsocket\b", r"\brequests\b",
    r"\burllib\b", r"\bhttp\b",
]

TIMEOUT_DEFAULT = 60
MAX_AUTO_INSTALL = 2


@dataclass
class ExecutionResult:
    """Résultat d'exécution Python."""
    success: bool
    stdout: str
    stderr: str
    code_executed: str
    timeout: bool = False
    libs_installed: list = None


# ── Helpers ──────────────────────────────────────────────────────
def check_security(code: str) -> str | None:
    """Vérifie qu'aucun pattern interdit n'est dans le code."""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, code):
            return f"Pattern interdit : {pattern}"
    return None


def detect_missing_module(stderr: str) -> str | None:
    """Détecte le nom du module manquant dans une stderr."""
    match = re.search(r"No module named ['\"](\w+)['\"]", stderr)
    return match.group(1) if match else None


def install_library(lib: str) -> bool:
    """Installe une librairie dans l'env conda."""
    print(f"  │  → Installation auto : {lib}")
    result = subprocess.run(
        [PYTHON_EXE, "-m", "pip", "install", "--quiet", lib],
        capture_output=True,
        text=True,
        timeout=180,
    )
    return result.returncode == 0


def execute_script(code: str, timeout: int = TIMEOUT_DEFAULT) -> ExecutionResult:
    """Exécute un script Python isolé une fois."""
    security_issue = check_security(code)
    if security_issue:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"SecurityError: {security_issue}",
            code_executed=code,
        )
    
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        tmp_path = f.name
    
    try:
        # ── FIX UTF-8 Windows ──
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"  # force UTF-8 dans le subprocess
        env["PYTHONUTF8"] = "1"
        
        result = subprocess.run(
            [PYTHON_EXE, "-X", "utf8", tmp_path],  # -X utf8 force UTF-8
            capture_output=True,
            timeout=timeout,
            env=env,
            # ── Ne pas décoder automatiquement ──
            text=False,  # binary mode
        )
        
        # Décoder manuellement avec gestion d'erreur
        stdout = (result.stdout or b"").decode("utf-8", errors="replace")
        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
        
        return ExecutionResult(
            success=(result.returncode == 0),
            stdout=stdout,
            stderr=stderr,
            code_executed=code,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"TimeoutError: exécution > {timeout}s",
            code_executed=code,
            timeout=True,
        )
    except Exception as e:
        return ExecutionResult(
            success=False,
            stdout="",
            stderr=f"SubprocessError: {e}",
            code_executed=code,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

# ── API publique ─────────────────────────────────────────────────
def execute_python(code: str, timeout: int = TIMEOUT_DEFAULT) -> ExecutionResult:
    """
    Exécute du code Python avec auto-installation des libs manquantes.
    Tente jusqu'à MAX_AUTO_INSTALL installations avant d'abandonner.
    """
    libs_installed = []
    
    for tentative in range(MAX_AUTO_INSTALL + 1):
        result = execute_script(code, timeout)
        result.libs_installed = libs_installed
        
        # Succès
        if result.success:
            return result
        
        # Échec — vérifier si c'est un module manquant
        missing = detect_missing_module(result.stderr)
        if missing is None:
            return result  # autre erreur, abandon
        
        # Module manquant — vérifier whitelist
        if missing not in ALLOWED_LIBS:
            result.stderr += f"\n[SecurityError: '{missing}' n'est pas dans la whitelist]"
            return result
        
        # Installation
        if tentative >= MAX_AUTO_INSTALL:
            return result  # trop d'installations
        
        ok = install_library(missing)
        if not ok:
            result.stderr += f"\n[InstallError: impossible d'installer '{missing}']"
            return result
        
        libs_installed.append(missing)
        # Retry automatique
    
    return result


# ── Test direct ──────────────────────────────────────────────────
if __name__ == "__main__":
    test_code = """
import numpy as np
from scipy.optimize import fsolve

def equation(x):
    return x**2 - 4

x = fsolve(equation, 1)
print(f"Solution : x = {x[0]:.4f}")
"""
    result = execute_python(test_code)
    print(f"Success : {result.success}")
    print(f"Stdout  : {result.stdout}")
    print(f"Stderr  : {result.stderr}")