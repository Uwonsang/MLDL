import datasets
import utils
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset_odn = datasets.C100Dataset('../dataset/cifar100_nl/data/cifar100_nl.csv')
[data_train_x, data_train_y, data_test_x, data_test_y] = dataset_odn.getDataset()

batch_size = 1
# Test the model
model_path = '../saved/CNN_pre/best_model.pt'
model = utils.SimpleCNN().to(device)
check_point = torch.load(model_path)
state_dict = check_point['net']
model.load_state_dict(state_dict)

print('Start test..')
model.eval()
with torch.no_grad():
    total_test_step = int(len(data_test_x))

    predicted_list = []

    for i in tqdm(range(total_test_step)):
        imgs, labels = dataloader(data_test_x, data_test_y, i, batch_size, mode='test').getload()
        imgs = imgs.to(device, dtype=torch.float)
        outputs = model(imgs)

        _, predicted = torch.max(outputs, 1)    # max()를 통해 최종 출력이 가장 높은 class 선택
        predicted = predicted.detach().cpu()
        predicted_list.append(predicted.numpy())

    predicted_list = np.array(predicted_list).flatten()

    ##replace origin_name
    data = pd.read_csv('../dataset/cifar100_nl/data/cifar100_nl.csv', names=['path', 'class'], header=None)
    class_name = np.unique(data['class'][:49999])
    origin_class = {key: value for key, value in enumerate(class_name)}

    ### To submit kaggle
    test_list = pd.read_csv('../dataset/kaggle.csv')
    kaggle_submission = pd.DataFrame({'Id': test_list['Id'], 'Category': predicted_list})
    kaggle_submission = kaggle_submission.replace(origin_class)
    kaggle_submission.to_csv('../dataset/kaggle_submission.csv', index=False)
