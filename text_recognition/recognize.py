import easyocr
import os

# 글자에 따라 차량의 용도를 다음과 같이 분류하고 있었다.
# 1. 일반 자가용, 비사업용 차량
# 가/나/다/라/마/거/너/더/러/머/버/서/어/저/고/노/도/로/모/보/소/오/조/구/누/두/루/무/부/수/우/주
# 2. 일반 사업용(택시, 버스)
# 아/바/사/자
# 3. 렌터카 (법인 또는 대여)
# 하/허/호
# 4. 택배용
# 배
# https://post.naver.com/viewer/postView.nhn?volumeNo=15128707&memberNo=40864363


def korean_recog(img):
    string =''
    reader = easyocr.Reader(
        lang_list=['ko'],
        gpu=False,
        detector='./craft_mlt_25k.pth',
        recognizer='./korean_g2.pth',
        download_enabled=False
    )
    # Make sure that is only recognizes certain characters
    result = reader.readtext(img, detail=0, allowlist='0123456789가나다라마거너더러머버서어저고노도로모보소오조구누두루무부수우주아바사자하허호배서울경기인천강원충남대전충북부산울대구경북남전광제')
    for i in result:
        string += i
    return string


# korean_recog('../detection/crop_black8361.jpg')  #126차 2861

num = 0
for file in sorted(os.listdir("../data_generate_license/generated_plate/test_plate/")):
    num += 1
    file_name = "../data_generate_license/generated_plate/test_plate/" + file
    # plate = cv2.imread(file_name)
    print(file)
    print(korean_recog(file_name))




