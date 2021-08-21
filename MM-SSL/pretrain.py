from dataloader.load_dataset import load_data

def train():
    dataloader = load_data()
    for batch in dataloader:
        pc0 = batch['pc0']
        pc1 = batch['pc1']
        img0 = batch['img0']
        img1 = batch['img1']
        d0 = batch['d0']
        d1 = batch['d1']
        #print(pc0)
        #print(pc1)

if __name__=="__main__":
    train()

