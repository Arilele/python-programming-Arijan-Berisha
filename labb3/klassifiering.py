def classify_point(x, y, k, m):
    return 1 if y > (k * x + m) else 0

def main():
    # Ta in värden från användaren
    try:
        x = float(input("Ange x-värdet för punkten: "))
        y = float(input("Ange y-värdet för punkten: "))
        k = float(input("Ange lutningen (k) för linjen: "))
        m = float(input("Ange skärningen (m) för linjen: "))
        
        # Klassificera punkten
        label = classify_point(x, y, k, m)
        
        # Skriv ut resultatet
        if label == 1:
            print(f"Punkten ({x}, {y}) ligger ovanför/höger om linjen y = {k}x + {m}.")
        else:
            print(f"Punkten ({x}, {y}) ligger under/vänster om linjen y = {k}x + {m}.")
    except ValueError:
        print("Felaktig inmatning! Se till att ange numeriska värden.")

if __name__ == "__main__":
    main()
