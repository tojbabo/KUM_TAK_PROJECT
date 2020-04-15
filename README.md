# KEUM_TAK_PROJECT
프로젝트 속성 - 구성 속성 - 디버깅 - 환경 <PATH=C:\Temp\opencv\build\x64\vc15\bin;%PATH%> 내용을 추가해 주어야 한다.
#
# - 0415
추가 보완 - UI에 출력되는 촬영본 전송하기 ver.3
 - 쓰레드에서 개체에 접근하는 것이 문제인가 싶어서 Dispatcher 를 통해 접근 [실패]
    ( 메시지 수신해서 메시지 박스에 출력하는 경우는 이것이 먹힘 )
 - 오히려 쓰레드에서 잘못된 개체 접근이라는 에러 발생.
 - 접근법 자체는 문제가 없어보이고 에러도 발생하지 않음. 그런데 출력이 안됨[해결]
    --> 엄연히 따지면 쓰레드가 UI객체에 접근을 못한것이 맞음.
        new를 통해 동적할당된 메모리는 생성한 쓰레드만 접근이 가능. 다른 쓰레드는 접근 불가. ....1
        UI 객체는 일반적인 방식으로 접근 불가. Dispatcher라는 또다른 외부 쓰레드를 통해 접근 ....2
        1 + 2 의 이유로 같은 Dispatcher에서 메모리 동적할당과 작업을 동시에 진행해야함. .. 성공 .
#
# - 0414
추가 보완 - UI에 출력되는 촬영본 전송하기 ver.2
 - C#에서 촬영한 이미지 C++ 에서 출력하기
 -> 두개의 다른 Mat 객체를 통일화 시키기 (해결)
1. C++에서 촬영한걸 C#에 가져오기 -> IntPtr 변수 사용한 방법 -> 되긴 하나 capture를 새로 실행해야해서 느림
2. C#에서 촬영한걸 C++에 가져가기 -> IntPtr을 byte[]로, byte[]를 다시 Mat(C++)로 -> 성공 속도 빠름

 - 그러나 기본적으로 쓰레드에서 UI에 출력이 안됨
#
# - 0413
추가 보완 - UI에 출력되는 촬영본 전송하기 ver.1
 - c++에서 소켓 생성, 연결해서 받아오기 (성공)
 - c#에서 촬영, c++로 소켓 아이디와 mat 보내서 전송 (실패)
 - 소켓 생성을 밖에서 한번 생성하고 쓰레드 에서 또 생성하면 작동되는데
   밖에서 생성안하고 쓰레드에서만 생성시 에러 발생 (이유 모름)

 - 위의 문제가 해결되었다고 가정 했을때, 동작시
    "System.InvalidOperationException: 이 형식의 계층 구조에 ComVisible(false) 부모가 있으므로 IDispatch 또는 클래스 인터페이스에 대한 QueryInterface를 호출할 수 없습니다."  에러 발생
 - c# wpf ui의 변수에 쓰레드가 접근하는데 뭔가 문제가 있는듯 함
 - 혹은 c#과 c++에서 취급하는 MAT 클래스가 다른듯함. 캐스팅 할 방법을 찾아야할 것 같음.
#
# - 0412
코드 수정
1. Token 관련 함수 이름 변경 및 기능 확실히설정

NEXT - Mat을 객체에 출력 -> DLL에서는 객체 접근 불가 -> 전송만 dll에서 하려니 socket descriptor를 넘겨주기 어려움
       -> dll에서  socket(descriptor)생성하여 받아옴, 이걸 촬영한 mat과 함께 보내면 어떨지 
#
# - 0410
디버그
1. 원래 되던 영상 송수신이 안됨. 이유를 찾아보아야 할것.
2. 원인 - 사용자 이름 설정때문, 정확한 이유는 모르나 채팅을 한번 보내서 USER NAME 을 한번 보내야 정상 작동됨
3. ^USER - 보내는 과정에서 \n이 안들어가서 에러 발생. -> 좀더 알아봐야할듯
#
# - 0225
채팅 보완
1. 연결 시에 이름을 설정할 수 있도록 구현

NEXT - c++ 과 c# 사이에 영상 출력 해결
#
# - 0224
UI 개선
1. 채팅 임시 구현
2. 이미지 사이즈 640 X 480 --> 320 X 240 으로 조정

NEXT - 다른 유저의 이미지를 받게 될 경우, 윈도우 위에 출력해야 하는데 dll(c++)과 wpf(c#) 간에 어떻게 주고받을지 알아내야함
#
# - 0218
통신간에 발생하는 문제점 개선
1. tcp 통신간에 버퍼가 끊기는 문제점 개선
   ( 데이터의 말단에  '\n'으로 정의하여 말단을 만나기 전까진 데이터를 수신했다고 생각하지 않도록 함)
2. 서버 측에선 아직 미구현 

NEXT - RECV Thread 구현
#
# - 0214
SEND THREAD 완벽구현
1. DLL 내에서 thread생성하여 send하는 기능 구현
2. IP와 PORT를 넘겨주어 thread내에서 socket을 생성하여 전송하도록 함.

NEXT - RECV thread 구현
#
# - 0213
dll을 thread로 구동
1. c++로 작성하여 dll로 만든 모듈을 thread로 구동함을 확인
2. 인자 넘겨주는 것 또한 정상적으로 동작됨을 확인

NEXT - send thread 구현 -> socket을 만들어서 인자로 넘겨줄지, socket생성 과정부터 thread에서 처리할지 결정해야함.
#
# - 0211
WPF 환경에서 dll 연동
1. sscanf의 부재점을 단순히 인덱싱 하여 substring하는 방식으로 해결
   sprintf는 그냥 문자열 합치기로 하면 될 것 같음.
2. 스레드 생성까지 무리 없이 해결 - 인자 넘겨주는 것은 아직 못함
3. 촬영한 영상을 전송하기 위해 처리 과정이 필요(OPENCV),
   but, C# 자체가 문법이 달라서 진행에 무리가 존재 --> C++ dll을 만들어서 참조하기로 결정
   dll을 만들어서 참조는 성공. (프로젝트\bin\release / debug에 dll 파일 추가)

NEXT - 이미지 처리하는 함수를 dll에 담아야 함. 
		이미지를 주고 socket에 보낼 수 있도록 변환시킨 데이터를 리턴할지, 혹은
		socket까지 넘겨주어서 전송시킬 수 있도록 할 지 결정해야함.
#
# - 0210
WPF 환경 구축 - MFC에서 바꿀 예정 
1. 기본적인 TCP 연결 확인
2. opencv 이미지 촬영 상태 확인

NEXT - 기본적인 문법자체가 조금 달라서 수정 필요 (sscanf / sprintf 등)
		서버 쪽과 통신에 있어서 제어 메시지를 조금 수정해야함
#
# - 0114
MFC 환경 구축 (Opencv / WInsock 등)
1. 기본적인 TCP 연결 상태 확인
2. opencv 이미지 촬영 상태 확인
3. udp 영상 통신 확인 전송은 잘 됨
#