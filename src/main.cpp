#include <Arduino.h>
#include <ESP32Servo.h>

static const int SERVO1_PIN = 18; // Trapdoor
static const int SERVO2_PIN = 19; // Pemutar wadah sampah

static const int TRAPDOOR_CLOSED_ANGLE = 180;
static const int TRAPDOOR_OPEN_ANGLE = 0; // Berputar 180 derajat dari posisi awal 180
static const int BIN_HOME_ANGLE = 0; // Posisi netral wadah saat standby

static const unsigned long BIN_SETTLE_MS = 1000; // Jeda wadah berputar sebelum trapdoor buka
static const unsigned long TRAPDOOR_DROP_MS = 1500; // Waktu trapdoor terbuka untuk jatuhkan sampah
static const unsigned long TRAPDOOR_CLOSE_MS = 500; // Waktu trapdoor menutup rapat
static const unsigned long BIN_RESET_MS = 800; // Waktu wadah kembali ke posisi netral

Servo servo1;
Servo servo2;

enum SortState {
  STATE_IDLE,
  STATE_ROTATE_BIN,
  STATE_OPEN_TRAPDOOR,
  STATE_CLOSE_TRAPDOOR,
  STATE_RESET_BIN
};

SortState currentState = STATE_IDLE;
unsigned long stateStartTime = 0;
int currentBinAngle = BIN_HOME_ANGLE;
int currentTrapdoorAngle = TRAPDOOR_CLOSED_ANGLE;

void printStatus() {
  int s1 = currentTrapdoorAngle;
  int s2 = currentBinAngle;
  const char* stateName = "IDLE";
  if (currentState == STATE_ROTATE_BIN) stateName = "ROTATE_BIN";
  else if (currentState == STATE_OPEN_TRAPDOOR) stateName = "TRAPDOOR_OPEN";
  else if (currentState == STATE_CLOSE_TRAPDOOR) stateName = "TRAPDOOR_CLOSING";
  else if (currentState == STATE_RESET_BIN) stateName = "RESET_BIN";

  Serial.printf(">> [STATUS] Servo1(Trapdoor): %d deg | Servo2(Wadah): %d deg | State: %s\n", s1, s2, stateName);
}

void printHelp() {
  Serial.println("Perintah serial:");
  Serial.println("  sort <deg>      - Urutan pemilahan: putar wadah ke <deg>, buka trapdoor, tutup, lalu wadah kembali ke 0 deg");
  Serial.println("  pos1 <deg>      - Kontrol manual Servo 1 (Trapdoor) 0-180");
  Serial.println("  pos2 <deg>      - Kontrol manual Servo 2 (Wadah) 0-180");
  Serial.println("  trapdoor open   - Buka trapdoor ke 0 derajat");
  Serial.println("  trapdoor close  - Tutup trapdoor ke 180 derajat");
  Serial.println("  home            - Parkir kedua servo ke posisi netral (Trapdoor 180, Wadah 0)");
  Serial.println("  status          - Baca sudut aktif Servo 1 dan Servo 2");
  Serial.println("  help            - Tampilkan menu perintah");
}

void startSortingSequence(int binAngle) {
  if (currentState != STATE_IDLE) {
    Serial.println(">> [BUSY] Pemilahan sebelumnya masih berjalan, perintah diabaikan.");
    return;
  }
  currentBinAngle = constrain(binAngle, 0, 180);
  servo2.write(currentBinAngle);
  currentState = STATE_ROTATE_BIN;
  stateStartTime = millis();
  Serial.printf(">> [SORT] Wadah (Servo 2) memutar ke %d derajat.\n", currentBinAngle);
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toLowerCase();

  if (cmd.length() == 0) return;

  if (cmd.startsWith("sort ") || cmd.startsWith("pos ")) {
    int spaceIdx = cmd.indexOf(' ');
    int target = cmd.substring(spaceIdx + 1).toInt();
    startSortingSequence(target);
  }
  else if (cmd.startsWith("pos1 ") || cmd.startsWith("servo1 ")) {
    int spaceIdx = cmd.indexOf(' ');
    int angle = constrain(cmd.substring(spaceIdx + 1).toInt(), 0, 180);
    currentTrapdoorAngle = angle;
    servo1.write(angle);
    Serial.printf(">> [MANUAL] Servo 1 (Trapdoor) diatur ke %d derajat.\n", angle);
  }
  else if (cmd.startsWith("pos2 ") || cmd.startsWith("servo2 ")) {
    int spaceIdx = cmd.indexOf(' ');
    int angle = constrain(cmd.substring(spaceIdx + 1).toInt(), 0, 180);
    currentBinAngle = angle;
    servo2.write(angle);
    Serial.printf(">> [MANUAL] Servo 2 (Wadah) diatur ke %d derajat.\n", angle);
  }
  else if (cmd == "trapdoor open") {
    currentTrapdoorAngle = TRAPDOOR_OPEN_ANGLE;
    servo1.write(TRAPDOOR_OPEN_ANGLE);
    Serial.printf(">> [ACTION] Trapdoor dibuka (%d derajat).\n", TRAPDOOR_OPEN_ANGLE);
  }
  else if (cmd == "trapdoor close") {
    currentTrapdoorAngle = TRAPDOOR_CLOSED_ANGLE;
    servo1.write(TRAPDOOR_CLOSED_ANGLE);
    Serial.printf(">> [ACTION] Trapdoor ditutup (%d derajat).\n", TRAPDOOR_CLOSED_ANGLE);
  }
  else if (cmd == "home" || cmd == "reset") {
    currentTrapdoorAngle = TRAPDOOR_CLOSED_ANGLE;
    currentBinAngle = BIN_HOME_ANGLE;
    servo1.write(TRAPDOOR_CLOSED_ANGLE);
    servo2.write(BIN_HOME_ANGLE);
    currentState = STATE_IDLE;
    Serial.printf(">> [ACTION] Posisi netral: Trapdoor %d deg | Wadah %d deg.\n", TRAPDOOR_CLOSED_ANGLE, BIN_HOME_ANGLE);
  }
  else if (cmd == "status" || cmd == "read") {
    printStatus();
  }
  else if (cmd == "help" || cmd == "?") {
    printHelp();
  }
  else {
    Serial.printf(">> [UNKNOWN] Perintah '%s' tidak dikenali. Ketik 'help'.\n", cmd.c_str());
  }
}

void updateSortingFSM() {
  if (currentState == STATE_IDLE) return;

  unsigned long elapsed = millis() - stateStartTime;

  if (currentState == STATE_ROTATE_BIN) {
    if (elapsed >= BIN_SETTLE_MS) {
      currentTrapdoorAngle = TRAPDOOR_OPEN_ANGLE;
      servo1.write(TRAPDOOR_OPEN_ANGLE);
      currentState = STATE_OPEN_TRAPDOOR;
      stateStartTime = millis();
      Serial.printf(">> [SORT] Wadah stabil, trapdoor (Servo 1) membuka ke %d derajat.\n", TRAPDOOR_OPEN_ANGLE);
    }
  }
  else if (currentState == STATE_OPEN_TRAPDOOR) {
    if (elapsed >= TRAPDOOR_DROP_MS) {
      currentTrapdoorAngle = TRAPDOOR_CLOSED_ANGLE;
      servo1.write(TRAPDOOR_CLOSED_ANGLE);
      currentState = STATE_CLOSE_TRAPDOOR;
      stateStartTime = millis();
      Serial.printf(">> [SORT] Sampah jatuh, trapdoor (Servo 1) menutup kembali ke %d derajat.\n", TRAPDOOR_CLOSED_ANGLE);
    }
  }
  else if (currentState == STATE_CLOSE_TRAPDOOR) {
    if (elapsed >= TRAPDOOR_CLOSE_MS) {
      currentBinAngle = BIN_HOME_ANGLE;
      servo2.write(BIN_HOME_ANGLE);
      currentState = STATE_RESET_BIN;
      stateStartTime = millis();
      Serial.printf(">> [SORT] Trapdoor tertutup rapat, wadah (Servo 2) kembali ke posisi netral (%d derajat).\n", BIN_HOME_ANGLE);
    }
  }
  else if (currentState == STATE_RESET_BIN) {
    if (elapsed >= BIN_RESET_MS) {
      currentState = STATE_IDLE;
      Serial.println(">> [SORT] Selesai. Sistem siap untuk sampah berikutnya.");
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);

  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);

  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.attach(SERVO2_PIN, 500, 2400);

  servo1.write(TRAPDOOR_CLOSED_ANGLE);
  servo2.write(currentBinAngle);

  Serial.println("\nESP32 IoTong Dual Servo Controller siap.");
  printHelp();
  printStatus();
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    handleCommand(input);
  }

  updateSortingFSM();
}
