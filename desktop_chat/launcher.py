"""Windows desktop owner for an isolated Edge window, chat server and Ollama."""
import ctypes
import json
import os
import subprocess
import socket
import sys
import time
import tkinter as tk
from tkinter import messagebox
import urllib.request
from pathlib import Path
from runtime import load_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'work/desktop_chat.json'


def main():
    # Per-session singleton. Closing an existing instance releases this mutex.
    mutex = ctypes.windll.kernel32.CreateMutexW(None, False, 'Local\\NntuDesktopChat')
    if ctypes.windll.kernel32.GetLastError() == 183:
        ctypes.windll.user32.MessageBoxW(None, 'Чат НГТУ уже запущен. Откройте его окно на панели задач.', 'НГТУ', 64)
        return
    config = load_config()
    # Do not accidentally attach to another user's already running runtime.
    for port in (config['chat_port'], config['ollama_port']):
        with socket.socket() as probe:
            try: probe.bind(('127.0.0.1', port))
            except OSError:
                ctypes.windll.user32.MessageBoxW(None, f'Порт {port} занят. Закройте предыдущий локальный чат.', 'НГТУ', 48)
                return
    window = tk.Tk()
    window.title('Чат НГТУ')
    window.geometry('470x210')
    window.configure(bg='#f3f7f9')
    window.resizable(False, False)
    icon = ROOT / 'desktop_chat/nntu.ico'
    if icon.exists(): window.iconbitmap(str(icon))
    tk.Label(window, text='Чат НГТУ', font=('Segoe UI', 22, 'bold'), bg='#f3f7f9', fg='#0072bc').pack(pady=(22, 7))
    status = tk.StringVar(value='Загружаю модели и базу знаний…')
    tk.Label(window, textvariable=status, font=('Segoe UI', 11), bg='#f3f7f9', wraplength=430).pack()
    processes, logs = [], []
    closed = False
    browser = None
    started = time.monotonic()
    env = os.environ.copy()
    env.update(PYTHONPATH=config['dependencies'], USE_TF='0', USE_FLAX='0',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
               OLLAMA_MODELS=config['ollama_models'], OLLAMA_HOST=f"127.0.0.1:{config['ollama_port']}")

    def spawn(command, name):
        log = (ROOT / 'work' / (name + '.log')).open('ab', buffering=0)
        logs.append(log)
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
        processes.append(process)
        return process

    def close():
        nonlocal closed
        if closed: return
        closed = True
        status.set('Останавливаю чат и освобождаю память…')
        window.deiconify()
        window.update_idletasks()
        for process in reversed(processes):
            if process.poll() is None:
                subprocess.run(['taskkill.exe', '/PID', str(process.pid), '/T', '/F'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               creationflags=subprocess.CREATE_NO_WINDOW)
                try: process.wait(timeout=10)
                except subprocess.TimeoutExpired: pass
        for log in logs: log.close()
        ctypes.windll.kernel32.CloseHandle(mutex)
        window.destroy()

    tk.Button(window, text='Закрыть и остановить', command=close, font=('Segoe UI', 11),
              bg='#0072bc', fg='white', relief='flat', padx=15, pady=7).pack(pady=19)
    window.protocol('WM_DELETE_WINDOW', close)

    def poll():
        nonlocal browser
        if closed: return
        if browser is not None:
            if browser.poll() is not None: return close()
        else:
            if any(p.poll() is not None for p in processes):
                messagebox.showerror('Чат НГТУ', 'Не удалось запустить чат. Журналы сохранены в папке work (desktop_*.log).')
                return close()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{config['chat_port']}/health", timeout=.3) as response:
                    ready = json.load(response).get('ready')
                if ready:
                    browser = spawn([config['browser'], f"--app=http://127.0.0.1:{config['chat_port']}/",
                                     '--user-data-dir=' + str(ROOT / 'work/desktop_edge_profile'),
                                     '--no-first-run', '--no-default-browser-check', '--disable-background-mode'], 'desktop_browser')
                    window.withdraw()
            except Exception:
                status.set(f'Загружаю модели и базу знаний… {int(time.monotonic()-started)} с')
                if time.monotonic()-started > 600:
                    messagebox.showerror('Чат НГТУ', 'Загрузка заняла больше 10 минут. Проверьте журналы в work.')
                    return close()
        window.after(1000, poll)

    try:
        spawn([config['ollama'], 'serve'], 'desktop_ollama')
        spawn([config['python'], '-u', str(ROOT / 'desktop_chat/server.py')], 'desktop_server')
        window.after(1000, poll)
        window.mainloop()
    finally:
        if not closed: close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        ctypes.windll.user32.MessageBoxW(None, 'Не удалось запустить чат: ' + str(error), 'НГТУ', 16)
