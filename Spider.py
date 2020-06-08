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
        'Cookie': 'guider_quick_search=on; listStyle=bigPhoto; is_searchv2=1; accessID=20200604232736840404; _gscu_1380850711=91496892rotxk867; save_jy_login_name=18616361446; stadate1=248684527; myloc=31%7C3109; myage=30; mysex=m; myuid=248684527; myincome=30; user_attr=000000; pop_sj=0; skhistory_f=a%3A2%3A%7Bi%3A1591497298%3Bs%3A6%3A%22%E4%B8%8A%E6%B5%B7%22%3Bi%3A1591284471%3Bs%3A3%3A%22%E7%94%B7%22%3B%7D; SESSION_HASH=02b5a9f52d722638a028c547f9397fe62c010dcf; user_access=1; COMMON_HASH=2e7d0dcddcbb3dcd46100bcc400a1df6; sl_jumper=%26cou%3D17%26omsg%3D0%26dia%3D0%26lst%3D2020-06-08; last_login_time=1591621109; upt=Z2l0icRJXf7I9lREUQVcask-2vaaCR0OaV7QKdOUftz-IlYWutInsReAkDC1mWu%2Avm1MXL5PRrgVAoT267RQHQU.; PROFILE=249684527%3Ajh%3Am%3Aimages1.jyimg.com%2Fw4%2Fglobal%2Fi%3A0%3A%3A1%3Azwzp_m.jpg%3A1%3A1%3A50%3A10%3A3.0; PHPSESSID=7093a51ea947697cd96f0a6515020472; pop_avatar=1; main_search:249684527=%7C%7C%7C00; RAW_HASH=X9OJhlcnfHRJD1XkIk9IlCL83sKDl84wUlrzNM-bn9cSMm7cjso2fgk3MM0HowINKPYGTGYiKyvUSY3pp-OMHFyUBxol9nWjswIDEfMMTWfgojk.; pop_time=1591621173301'
    }
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()
        r.encoding = 'unicode_escape'
        # r.encoding=r.apparent_encoding
        print(r.url)
        str = r.text.replace('\n', '').replace('\r', '').replace("  伀筶 疼 Уán茚", "XX")  # 用于处理text中的乱码，但是数量太多了，使用try直接跳过
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
    for pageNo in range(1, 2000):  # 最后的值即为爬取的总页数，手动修改数值即可。总页数可以在printhtml的pageTotal中得到。
        url = 'http://search.jiayuan.com/v2/search_v2.php?key=&sex=f&stc=2:18.31,3:155.175,23:1&sn=default&sv=1&p={0}&f=select'.format(pageNo) #sex=f即为女性，sex=m即为男性。stc=2:18.31表示年龄18-31，3:155.175表示身高范围, 23:1表示有照片
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
