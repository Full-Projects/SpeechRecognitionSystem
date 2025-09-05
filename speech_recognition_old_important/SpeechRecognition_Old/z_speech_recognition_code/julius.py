# coding:utf-8
import re
import socket
import subprocess
import time


HOST = "127.0.0.1"
PORT = 5000
JULIUS_DIR = r"C:\Users\User\Desktop\julius\4.4\julius-4.4.2-win32bin\\"
#r'C:\Program Files (x86)\julius-4.4.2-win32bin\\'

re_word = re.compile('WORD="([^"]+)"')

def main():
    p = subprocess.Popen([JULIUS_DIR + "julius.exe", "-C", r"C:\Users\User\Desktop\julius\model\ENVR-v5.4.Dnn.Bin\julius.jconf", "-dnnconf", r"C:\Users\User\Desktop\julius\model\ENVR-v5.4.Dnn.Bin\dnn.jconf", "-module", ">", "nul"],
        stdout=subprocess.PIPE, shell=True)
    time.sleep(4)
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((HOST, PORT))
    print('SPEECH START')

    try:
        data = ""
        while 1:
            if "</RECOGOUT>" in data:
                words = ''
                for word in filter(bool, re_word.findall(data)):
                    words += word
                if words:
                    print(words)
                data = ""
            else:
                data = data + client.recv(1024)
    except KeyboardInterrupt:
        print("KeyboardInterrupt occured.")
        p.kill()
        client.close()

if __name__ == "__main__":
    main()
