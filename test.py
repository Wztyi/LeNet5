import torch
import torch.utils.data as Data
from torchvision import transforms
from torchvision.datasets import FashionMNIST
from model import LeNet


# 数据预处理
def test_data_process():
    datasets = FashionMNIST(
        root="./data", train=False, transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]), download=True
    )
    test_dataloader = Data.DataLoader(dataset=datasets, batch_size=1, shuffle=True, num_workers=0)

    return test_dataloader


def test_model_process(model, test_dataloader):
    # 设定训练所用到的设备, 有GPU则用GPU不然就用CPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 模型放入设备当中
    model = model.to(device)

    # 初始化参数
    test_corrects = 0.0
    # 测试数据总数
    test_num = 0

    # 推理不需要涉及到反向传播,只需要前向传播即可 不需要更新参数
    with torch.no_grad():
        for test_data_x, test_data_y in test_dataloader:
            # 将特征放入到训练设备中
            test_data_x = test_data_x.to(device)
            # 将标签放入到训练设备中
            test_data_y = test_data_y.to(device)
            # 设置为评估模式
            model.eval()
            # 前向传播的过程,输入为测试数据集,输出为输出每个样本的预测值
            output = model(test_data_x)
            # 查找每一行中最大值对应的行标
            pre_label = torch.argmax(output, dim=1)
            # 如果预测正确,则准确度test_corrects + 1
            test_corrects += torch.sum(pre_label == test_data_y.data)
            # 将所有的测试样本进行累加
            test_num += test_data_x.size(0)
    # 测试准确率
    test_acc = test_corrects.double().item() / test_num
    print("测试的准确率为:", test_acc)


if __name__ == "__main__":
    # 加载模型
    model = LeNet()
    model.load_state_dict(torch.load("./runs/best_model.pth"))
    # 加载测试数据
    test_dataloader = test_data_process()
    # 加载模型测试的函数
    test_model_process(model, test_dataloader)

    # 设定测试所用到设备，有GPU的用GPU，没有则用CPU来进行测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    with torch.no_grad():
        for b_x, b_y in test_dataloader:
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            # 设置为验证模型
            model.eval()
            output = model(b_x)
            pre_label = torch.argmax(output, dim=1)
            result = pre_label.item()
            label = b_y.item()

            print("预测值:", classes[result], "--------", "真实值:", classes[label])
