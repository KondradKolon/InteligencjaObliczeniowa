from datetime import date
import math

imie=input("podaj imie: ")
rok=int(input("podaj rok urodzenia: "))
miesiac=int(input("podaj miesiac urodzenia: "))
dzien=int(input("podaj dzien urodzenia: "))

date1=date.today()
date2=date(rok,miesiac,dzien)
ileDni=(date1-date2).days
print(ileDni)


def FizycznaFala(t):
    result = math.sin((2 * math.pi / 23) * t)
    return result


def EmocjonalnaFala(t):
    result = math.sin((2 * math.pi / 28) * t)
    return result


def IntelektualnaFala(t):
    result = math.sin((2 * math.pi / 33) * t)
    return result

print(f"Twoja Fizyczna fala: {FizycznaFala(ileDni)}")

print(f"Twoja Emocjonalna  fala: {EmocjonalnaFala(ileDni)}")

print(f"Twoja Intelektualna fala: {IntelektualnaFala(ileDni)}")


def czyjutrobedziejepiej(t,typ):
    if typ(t+1) > typ(t):
        print(f"Jutro bedzie lepsza {typ}")
    else:
        print("jutro kys")

if (FizycznaFala(ileDni) > 0.5):
    print("Gratulacje dobrego wyniku Fizycznego")


if (EmocjonalnaFala(ileDni) > 0.5):
    print("Gratulacje dobrego wyniku Emocjonalnego")


if (IntelektualnaFala(ileDni) > 0.5):
    print("Gratulacje dobrego wyniku Intelektualnego")

#

if (FizycznaFala(ileDni) < -0.5):
    print("nie przejmuj sie wynikiem fizycznej fali to glupoty")
    czyjutrobedziejepiej(ileDni,FizycznaFala)


if (EmocjonalnaFala(ileDni) < -0.5):
    print("nie przejmuj sie wynikiem Emocjonalnej fali to glupoty")
    czyjutrobedziejepiej(ileDni,EmocjonalnaFala)

if (IntelektualnaFala(ileDni) < -0.5):
    print("nie przejmuj sie wynikiem Intelektualnej fali to glupoty")
    czyjutrobedziejepiej(ileDni,IntelektualnaFala)