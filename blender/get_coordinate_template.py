import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--posx",type=float)
parser.add_argument("--sizex",type=float)
parser.add_argument("--posy",type=float)
parser.add_argument("--sizey",type=float)
args=parser.parse_args()

def coordx(pos,dim):
    return (pos -6.5 +dim/2)/100

def coordy(pos,dim):
    return (pos -5.5 -dim/2)/100

print("Xc: ", coordx(args.posx,args.sizex), "\nYc: ", coordy(args.posy,args.sizey))