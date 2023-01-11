# Semantic Segmentation Competition CV-13

## 🕵️Members

<table>
    <th colspan=5>📞 TEAM 031</th>
    <tr height="160px">
        <td align="center">
            <a href="https://github.com/LimePencil"><img src="https://avatars.githubusercontent.com/u/71117066?v=4" width="100px;" alt=""/><br /><sub><b>신재영</b></sub></a>
        </td>
        <td align="center">
            <a href="https://github.com/sjz1"><img src="https://avatars.githubusercontent.com/u/68888169?v=4" width="100px;" alt=""/><br /><sub><b>유승종</b></sub></a>
        </td>
        <td align="center">
            <a href="https://github.com/SangJunni"><img src="https://avatars.githubusercontent.com/u/79644050?v=4" width="100px;" alt=""/><br /><sub><b>윤상준</b></sub></a>
        </td>
        <td align="center">
            <a href="https://github.com/lsvv1217"><img src="https://avatars.githubusercontent.com/u/113494991?v=4" width="100px;" alt=""/><br /><sub><b>이성우</b></sub></a>
        </td>
         <td align="center">
            <a href="https://github.com/0seob"><img src="https://avatars.githubusercontent.com/u/29935109?v=4" width="100px;" alt=""/><br /><sub><b>이영섭</b></sub></a>
        </td>
    </tr>
</table>

## 🗑️재활용 품목 분류를 위한 Semantic Segmentation
![image](https://user-images.githubusercontent.com/29935109/211736058-076bca36-c60e-41fd-b10b-ede511ebcc93.png)
>바야흐로 대량 생산, 대량 소비의 시대, 우리는 많은 물건이 대량으로 생산되고, 소비되는 시대를 살고 있습니다. 하지만 이러한 문화는 '쓰레기대란','매립지 부족'과 같은 여러 사회문제를 낳고 있습니다.

>분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다. 따라서 우리는 사진에서 쓰레기를 Segmentation하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위한 데이터셋으로는 배경, 일반 쓰레기, 플라스틱, 종이, 유리 등 11 종류의 쓰레기가 찍힌 사진 데이터셋이 제공됩니다. 여러분에 의해 만들어진 우수한 성능의 모델은 쓰레기장에 설치되어 정확한 분리수거를 돕거나, 어린아이들의 분리수거 교육 등에 사용될 수 있을 것입니다. 부디 지구를 위기로부터 구해주세요! 🌎

## 💾 Datasets
- 전체 이미지 개수 : 4091장
   - train : 3272장
   - test : 819장
- 11 class : Background, General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (512, 512)
- annotation format : COCO format


## 🗓️Timeline
![image](https://user-images.githubusercontent.com/29935109/211736165-d0f9ab28-3dd0-458d-8657-77577c5e6e97.png)


## 🧑‍💻Team Roles
><b>신재영</b>
>
>&nbsp;&nbsp;&nbsp;&nbsp;SMP와 MMSeg를 위한 baseline 작성
>
>&nbsp;&nbsp;&nbsp;&nbsp;Sweep Configuration, Hard-voting ensemble


> <b>유승종</b>
>
>&nbsp;&nbsp;&nbsp;&nbsp;U-Net, U-Net++, U-Net 3+ 모델 분석
>
>&nbsp;&nbsp;&nbsp;&nbsp;OpenCV를 활용한 전처리
>
>&nbsp;&nbsp;&nbsp;&nbsp;k-fold 구현


> <b>윤상준</b>
>
>&nbsp;&nbsp;&nbsp;&nbsp;Detectron2, MMsegmentation 환경 구성 및 학습
>
>&nbsp;&nbsp;&nbsp;&nbsp; 라이브러리 별 학습을 위한 데이터셋 구성 변경
>
>&nbsp;&nbsp;&nbsp;&nbsp; Wandb Sweep 환경 구성


> <b>이성우</b>
>
>&nbsp;&nbsp;&nbsp;&nbsp;Pseudo labeling 코드 작성
>
>&nbsp;&nbsp;&nbsp;&nbsp;MMsegmentation 환경 구성 및 학습
>
>&nbsp;&nbsp;&nbsp;&nbsp;EDA


> <b>이영섭</b>
>
>&nbsp;&nbsp;&nbsp;&nbsp;k-fold dataset 코드 작성
>
>&nbsp;&nbsp;&nbsp;&nbsp;mmsegmentation train 환경 구축
>
>&nbsp;&nbsp;&nbsp;&nbsp;mIoU metric 분석
>

## 🏔️Environments
### <img src="https://cdn3.emoji.gg/emojis/4601_github.png" alt="drawing" width="16"/>  GitHub
- 모든 코드들의 버전관리
- GitFlow를 이용한 효율적인 전략
- Issue를 통해 버그나 프로젝트 관련 기록
- PR을 통한 code review

### <img src="https://img.icons8.com/ios-filled/500/notion.png" alt="drawing" width="16"/> Notion
- 노션을 이용하여 실험결과등을 정리
- 회의록을 매일 기록하여 일정을 관리
- 가설 설정 및 결과 분석 등을 기록
- 캘린더를 사용하여 주간 일정 관리

### <img src="https://cdn.icon-icons.com/icons2/2699/PNG/512/atlassian_jira_logo_icon_170511.png" alt="drawing" width="16"/> Jira
- 발생하는 모든 실험의 진행상황 기록
- 로드맵을 통한 스케줄 관리
- 효율적인 일 분배 및 일관성 있는 branch 생성

### <img src="https://avatars.githubusercontent.com/u/26401354?s=200&v=4" alt="drawing" width="16"/> WandB
- 실험들의 기록 저장 및 공유
- 모델들의 성능 비교
- Hyperparameter 기록
- 총 672시간 기록

## ⚙️Requirements
```
Ubuntu 18.04.5 LTS
Intel(R) Xeon(R) Gold 5120 CPU @ 2.20GHz
NVIDIA Tesla V100-PCIE-32GB

conda install pytorch=1.7.1 cudatoolkit=11.0 torchvision -c pytorch  
pip install openmim  
mim install mmseg  
```
[Link To Installation Guide](https://github.com/boostcampaitech4lv23cv2/level2_semanticsegmentation_cv-level2-cv-13/issues, "Click to move issue page")

## 🎉Results🎉
>### Public LB : 11th (mAP 0.7502)
![image](https://user-images.githubusercontent.com/29935109/211736691-96c0fe1a-120e-4e2f-ab68-f8b931f971ba.png)
>### Private LB : 12th (mAP 0.7296)
![image](https://user-images.githubusercontent.com/29935109/211736663-6eb4e516-6615-477b-88ee-18893aded854.png)


## 📌Please Look at our Wrap-Up Report for more details
[![image](https://user-images.githubusercontent.com/62556539/200262300-3765b3e4-0050-4760-b008-f218d079a770.png)](https://www.notion.so/Segmentation-wrap-up-report-c6478ce7542c460888f1cc8a647ec395)
