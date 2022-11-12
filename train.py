from data import *
from foldasppunet import *
from torch import nn, optim
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

sumWriter_tr = SummaryWriter(log_dir= "log0.09/tr")
sumWriter_va = SummaryWriter(log_dir= "log0.09/va")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_path = "u0.09FDA.pth"
best_path = "best_0.09.pth"
train_path = "hecheng/train"
val_path = train_path.replace("train", "test")
save_path = "pred"
pi = 0
vpi=0
di=0
vdi=0
io=0
vio=0
mi = 555
if __name__ == "__main__":
    train_loader = DataLoader(FDA_dataset(train_path,"hecheng/train/tar", transform=transforms), batch_size=2, shuffle=False)
    val_loader = DataLoader(FDA_dataset(val_path,None, transform=transforms), batch_size=2, shuffle=False)
    print("number", len(train_loader))
    print("number", len(val_loader))
    print("data loader ok!")
    net = unet(3,1).to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print("successful load weight!")
    else:
        print("not successful load weight!")
    optimizer = optim.Adam(net.parameters())
    loss_fn = nn.BCELoss()
    epoch = 100

    for i in range(epoch):
        for j, (image, label) in enumerate(train_loader):
            losses = []

            image, label = image.to(device), label.to(device)
            pred = net(image)
            pred = torch.sigmoid(pred)
            loss = loss_fn(pred,label)
            loss_value = loss.data.cpu().numpy()
            losses.append(loss_value)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sumWriter_tr.add_scalar("train loss", loss.item(), i*len(train_loader)+j+1)
        losss = np.mean(losses)
        sumWriter_tr.add_scalar("epoch loss", losss, i)


        if i%20 == 0:
            torch.save(net.state_dict(), weight_path)
            print("save successfully!")

        net.eval()
        correct = 0
        corrects = 0
        pixel = 0
        pixels = 0
        dice = 0
        dices = 0
        with torch.no_grad():
            for (image, label) in train_loader:
                image, label = image.to(device), label.to(device)
                pred = torch.sigmoid(net(image))
                pred = (pred > 0.5)
                correct = (pred == label).sum()
                corrects += correct
                pixel = torch.numel(pred)
                pixels += pixel
                dice = (2 * (pred * label).sum()) / ((pred + label).sum() + 1e-8)
                dices += dice
            Pa = corrects / pixels * 100
            Dice = dices / len(train_loader)
            Iou = Dice / (2 - Dice)
            if pi<Pa:
                pi=Pa
            if di<Dice:
                di=Dice
            if io<Iou:
                io=Iou
            print("pa:", Pa)
            print("Dice:", Dice)
            print("Iou:", Iou)
            sumWriter_tr.add_scalar("Pa", Pa, i)
            sumWriter_tr.add_scalar("Dice", Dice, i)
            sumWriter_tr.add_scalar("Iou", Iou, i)


        for k, (image, label) in enumerate(val_loader):
            losses = []
            image, label = image.to(device), label.to(device)
            pred = net(image)
            pred = torch.sigmoid(pred)
            loss = loss_fn(pred, label)
            loss_value = loss.data.cpu().numpy()
            losses.append(loss_value)
            sumWriter_va.add_scalar("loss", loss.item(), i * len(val_loader) + k + 1)
        losss = np.mean(losses)
        sumWriter_va.add_scalar("epoch loss", losss, i)
        if losss<mi:
            mi = losss
            torch.save(net.state_dict(), best_path)
            print("save best!")

        net.eval()
        correct = 0
        corrects = 0
        pixel = 0
        pixels = 0
        dice = 0
        dices = 0
        with torch.no_grad():
            for (image, label) in val_loader:
                image, label = image.to(device), label.to(device)
                pred = torch.sigmoid(net(image))
                pred = (pred > 0.5)
                correct = (pred == label).sum()
                corrects += correct
                pixel = torch.numel(pred)
                pixels += pixel
                dice = (2 * (pred * label).sum()) / ((pred + label).sum() + 1e-8)
                dices += dice
            vPa = corrects / pixels * 100
            vDice = dices / len(val_loader)
            vIou = vDice / (2 - vDice)
            if vpi<vPa:
                vpi=vPa
            if vdi<vDice:
                vdi=vDice
            if vio<vIou:
                vio=vIou
            print("vpa:", vPa)
            print("vDice:", vDice)
            print("vIou:", vIou)
            sumWriter_va.add_scalar("Pa", vPa, i)
            sumWriter_va.add_scalar("Dice", vDice, i)
            sumWriter_va.add_scalar("Iou", vIou, i)
print("pi:",pi)
print("vpi:",vpi)
print("di:",di)
print("vdi:",vdi)
print("io:",io)
print("vio:",vio)
