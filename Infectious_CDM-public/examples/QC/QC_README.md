# 품질진단 어떻게하지?



# 지표 설정
    정확성, 완전성, 일관성, 유효성, 시기적절성, 기초통계
# 항목 설정
  **정형데이터 내용(ISO/IEC 25024)**
  <table>
    <tr>
        <th>번호</th>
        <th>기준</th>
        <th>심사항목명</th>
        <th>적용기준</th>
    </tr>
    <tbody>
        <tr>
            <td class="txtc">1</td>
            <td rowspan="2" class="txtc bb1"><strong>완전성</strong></td>
            <td class="txtc">데이터 값 완전성</td>
            <td>데이터 명세 등에 Notnull 조건이 있는 경우</td>
        </tr>
        <tr>
            <td class="txtc bb1">2</td>
            <td class="txtc bb1">데이터 레코드 완전성</td>
            <td class="bb1">반정형은 레코드를 정의할 수 있는 경우</td>
        </tr>
        <tr class="str">
            <td class="txtc">3</td>
            <td rowspan="4" class="txtc bb1"><strong>유효성</strong></td>
            <td class="txtc">구문 유효성</td>
            <td>구문정확성이 요구되는 경우 (ex. 도메인 규칙, 데이터 타입 등)</td>
        </tr>
        <tr class="str">
            <td class="txtc">4</td>
            <td class="txtc">의미 유효성</td>
            <td>해당 필드 또는 속성에 의미상으로 유효한 리스트가 존재하는 경우</td>
        </tr>
        <tr class="str">
            <td class="txtc">5</td>
            <td class="txtc">범위 유효성</td>
            <td>명세서 등에 범위(숫자)가 정의된 경우 (예: 최대값, 최소값이 적용 가능한 수치)</td>
        </tr>
        <tr class="str">
            <td class="txtc bb1">6</td>
            <td class="txtc bb1">관계 유효성</td>
            <td class="bb1">명세서 등에 의미론적 업무규칙이 정의된 경우 (예: 성별과 주민등록번호의 뒤 첫자리)</td>
        </tr>
        <tr>
            <td class="txtc bb1">7</td>
            <td class="txtc bb1"><strong>일관성</strong></td>
            <td class="txtc bb1">참조 무결 일관성</td>
            <td class="bb1">명세서 등에 참조 무결성이 정의된 경우</td>
        </tr>
        <tr class="str">
            <td class="txtc bb1">8</td>
            <td class="txtc bb1"><strong>유효성</strong></td>
            <td class="txtc bb1">데이터 값 정밀성</td>
            <td class="bb1">명세서 등에 정밀도가 정의된 경우<br>※ 반정형의 경우 단위와 자리수를 고려하여 측정 (예: 자리수-소수점, 시간 등)</td>
        </tr>
        <tr>
            <td class="txtc">9</td>
            <td rowspan="2" class="txtc bb1"><strong>일관성</strong></td>
            <td class="txtc">데이터 포맷 일관성</td>
            <td>명세서 등에 데이터 포맷이 정의되어 있고, (예: 날짜표시형식) 2개 이상의 유사한 특성 컬럼이 존재해야 함</td>
        </tr>
        <tr>
            <td class="txtc bb1">10</td>
            <td class="txtc bb1">공통 어휘 일관성</td>
            <td class="bb1">명세서 등에 데이터 사전(공통어휘) 정의된 경우</td>
        </tr>
        <tr class="str">
            <td class="txtc">11</td>
            <td rowspan="2" class="txtc bb1"><strong>정확성</strong></td>
            <td class="txtc">메타 데이터 정확성</td>
            <td>요구사항 명세서에 메타 데이터에 제공되어야 하는 정보가 기록된 경우</td>
        </tr>
        <tr class="str">
            <td class="txtc bb1">12</td>
            <td class="txtc bb1">데이터 값 정확성</td>
            <td class="bb1">기대되는 데이터 아이템의 값이 정의된 경우(예: 업무규칙 등)</td>
        </tr>
        <tr>
            <td class="txtc bb1">13</td>
            <td class="txtc bb1"><strong>접근성</strong></td>
            <td class="txtc bb1">표준기반 데이터 접근성</td>
            <td class="bb1">표준, 협약 또는 규정이 존재하는 데이터의 경우</td>
        </tr>
        <tr class="str">
            <td class="txtc bb1">14</td>
            <td class="txtc bb1"><strong>유일성</strong></td>
            <td class="txtc bb1">데이터 값 유일성</td>
            <td class="bb1">명세서 내 값의 중복을 허용되지 않는 데이터 아이템이 존재하는 경우</td>
        </tr>
        </tbody>
  </table>
    ref: 데이터 품질진단<https://dq.tecel.kr/dqc/about.php>

# 코드 작성
# 결과 자료
    # 지표별 수치
    # 대시보드(엑셀)
