import time
import requests
import json
import pandas as pd


def fetchHtml(url):
    """
    获取网页内容。
    :param url: 网址
    :return: 网页内容
    :Cookie:用于登录网页
    """

    headers = {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        'Connection': 'keep-alive',
        'Cookie': 'guider_quick_search=on; listStyle=bigPhoto; is_searchv2=1; accessID=20200604232736840404; save_jy_login_name=18616361446; myuid=248684527; PHPSESSID=292211238ed6adc6bd205ada53857c67; SESSION_HASH=81103ed716a5b4be3ffe54903350a58bf9a67cd5; user_access=1; stadate1=248684527; myloc=31%7C3109; myage=30; mysex=m; myincome=30; COMMON_HASH=2e7d0dcddcbb3dcd46100bcc400a1df6; sl_jumper=%26cou%3D17%26omsg%3D0%26dia%3D0%26lst%3D2020-06-29; last_login_time=1593440755; upt=mkMU7saWiTOeLsEJVmmvZ59rCSaqXI%2Atqim6%2AqzOJ6q2pfslHAk9dlQuiHt8k5-X76wKIVdPalSOg%2Aicsv9Ak-I.; user_attr=000000; main_search:249684527=%7C%7C%7C00; pop_avatar=1; PROFILE=249684527%3Ajh%3Am%3Aimages2.jyimg.com%2Fw4%2Fglobal%2Fi%3A0%3A%3A1%3Azwzp_m.jpg%3A1%3A1%3A50%3A10%3A3.0; RAW_HASH=tUsUpJ9DhPjhHfT391pxuGmGdeySHJOT5MZp7jlUzRB9Kn7JISEIsQiryeUdc8YaIaTRZIwiKzvKIXrunN15vYoXC2YcnNZKItfdrDa1jcXdPQo.; pop_time=1592746656382'
    }
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        # r.encoding = 'unicode_escape'
        # r.encoding=r.apparent_encoding
        print(r.url)
        str = r.text.replace("##jiayser##","")
        return str
    except requests.HTTPError as e:
        print(e)
        print("HTTPError")
    except requests.RequestException as e:
        print(e)
    except:
        print("Unknown Error !")


def parseHtml(html):
    """
    解析Html，并写入到CSV中
    :param html:
    :return:
    """
    try:
        s = json.loads(html)
        userinfo = []
        for key in s['userInfo']:
            blist = []
            uid = key['uid']
            nickname = key['nickname']
            sex = key['sex']
            age = key['age']
            work_location = key['work_location']
            height = key['height']
            education = key['education']
            matchCondition = key['matchCondition']
            marriage = key['marriage']
            income = key['income']
            shortnote = key['shortnote']
            image = key['image']

            blist.append(uid)
            blist.append(nickname)
            blist.append(sex)
            blist.append(age)
            blist.append(work_location)
            blist.append(height)
            blist.append(education)
            blist.append(matchCondition)
            blist.append(marriage)
            blist.append(income)
            blist.append(shortnote)
            blist.append(image)

            userinfo.append(blist)
            # print(nickname, age, work_location)
            #         #
            #         # print('---' * 20)

            # 写入到CSV中
            writeCSV(userinfo)
    except json.decoder.JSONDecodeError: #处理text中的乱码
        pass


def writeCSV(userinfo):
    dataframe = pd.DataFrame(userinfo)

    dataframe.to_csv('世纪佳缘_UserInfo.csv', encoding='utf-8-sig', mode='a', index=False, sep=',', header=False)


def dropDuplicates():  # 去重
    df = pd.read_csv('世纪佳缘_UserInfo.csv', encoding='utf-8-sig',
                     names=['uid', 'nickname', 'sex', 'age', 'work_location', 'height', 'education', 'matchCondition',
                            'marriage', 'income', 'shortnote', 'image'])
    datalist = df.drop_duplicates()
    datalist.to_csv('世纪佳缘_去重_UserInfo.csv', encoding='utf-8-sig', index=False, header=False)


if __name__ == '__main__':
    for pageNo in range(1, 200):  # 最后的值即为爬取的总页数，手动修改数值即可。总页数可以在printhtml的pageTotal中得到。
        # url = 'https://search.jiayuan.com/v2/search_v2.php?key=&sex=f&stc=2:18.31,3:155.175,23:1&sn=default&sv=1&p={0}&f=select'.format(pageNo) #sex=f即为女性，sex=m即为男性。stc=2:18.31表示年龄18-31，3:155.175表示身高范围, 23:1表示有照片
        url ='https://search.jiayuan.com/v2/search_v2.php?key=&sex=m&stc=2:18.25,3:165.190,23:1&sn=default&sv=1&p={0}&pt=7862&ft=off&f=select&mt=d'.format(pageNo)
        try:
            html = fetchHtml(url)
            # print(html)
            parseHtml(html)
        except:
            pass

        # 为了降低被封ip的风险，每爬100页便歇5秒。
        if pageNo % 100 == 99:
            time.sleep(5)

    for pageNo in range(1, 200):  # 最后的值即为爬取的总页数，手动修改数值即可。总页数可以在printhtml的pageTotal中得到。
        # url = 'https://search.jiayuan.com/v2/search_v2.php?key=&sex=f&stc=2:18.31,3:155.175,23:1&sn=default&sv=1&p={0}&f=select'.format(pageNo) #sex=f即为女性，sex=m即为男性。stc=2:18.31表示年龄18-31，3:155.175表示身高范围, 23:1表示有照片
        url ='https://search.jiayuan.com/v2/search_v2.php?key=&sex=m&stc=2:26.35,3:165.190,23:1&sn=default&sv=1&p={0}&pt=1783&ft=off&f=select&mt=d'.format(pageNo)
        try:
            html = fetchHtml(url)
            # print(html)
            parseHtml(html)
        except:
            pass

        # 为了降低被封ip的风险，每爬100页便歇5秒。
        if pageNo % 100 == 99:
            time.sleep(5)

    for pageNo in range(1, 200):  # 最后的值即为爬取的总页数，手动修改数值即可。总页数可以在printhtml的pageTotal中得到。
        # url = 'https://search.jiayuan.com/v2/search_v2.php?key=&sex=f&stc=2:18.31,3:155.175,23:1&sn=default&sv=1&p={0}&f=select'.format(pageNo) #sex=f即为女性，sex=m即为男性。stc=2:18.31表示年龄18-31，3:155.175表示身高范围, 23:1表示有照片
        url ='https://search.jiayuan.com/v2/search_v2.php?key=&sex=m&stc=2:36.45,3:165.190,23:1&sn=default&sv=1&p={0}&pt=14592&ft=off&f=select&mt=d'.format(pageNo)
        try:
            html = fetchHtml(url)
            # print(html)
            parseHtml(html)
        except:
            pass

        # 为了降低被封ip的风险，每爬100页便歇5秒。
        if pageNo % 100 == 99:
            time.sleep(5)

    dropDuplicates()
    print('Finish!')
