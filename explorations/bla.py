class T:
    def __init__(self) -> None:
        for l in range(3):
            vars(self)[f'layer_{l}'] = [l+2,l+3]

a = T()
for i in range(3):