import requests
import pandas as pd
import re


def avaterDownload():
    userData = pd.read_csv('世纪佳缘_去重_UserInfo.csv',
                           names=['uid', 'nickname', 'sex', 'age', 'work_location', 'height', 'education',
                                  'matchCondition', 'marriage', 'income', 'shortnote', 'image'])

    for line in range(len(userData)):
        try:
            url = userData['image'][line]
            img = requests.get(url).content
            nickname = re.sub(r"[\s+\.\!\/_,$%^*(+\"\'?|]+|[+——！，。？、~@#￥%……&*（）▌]+", "", userData['nickname'][line])
            filename = str(line) + '-' + nickname + '-' + str(userData['height'][line]) + '-' + str(
            userData['age'][line]) + '.jpg'

            with open('images/' + filename, 'wb') as f:
                f.write(img)
            print(filename + ' Succeeded')
        except:
            print(filename+' Failed')

    print("Finish!")


if __name__ == '__main__':
    avaterDownload()
