#include <Arduino.h>
#include <ESP32Servo.h>

// Definisi pin kontrol servo
static const int SERVO1_PIN = 18;
static const int SERVO2_PIN = 19;

Servo servo1;
Servo servo2;

// State Controller (Non-blocking)
bool isRunning = false;
int currentAngle = 0;
int sweepDirection = 1;          // +1 untuk naik (0->180), -1 untuk turun (180->0)
int speedDelayMs = 5;            // Delay per step dalam milidetik (makin kecil makin cepat)
unsigned long lastStepTime = 0;

void printHelp() {
  Serial.println("\n=============================================");
  Serial.println("   ESP32 Dual Servo Interactive Controller   ");
  Serial.println("=============================================");
  Serial.println("Pin terhubung:");
  Serial.println("  - Servo 1 -> GPIO 18 (D18)");
  Serial.println("  - Servo 2 -> GPIO 19 (D19)");
  Serial.println("\nPerintah yang bisa kamu ketik di Serial Monitor:");
  Serial.println("  start      -> Mulai gerakan kedua servo");
  Serial.println("  stop       -> Hentikan gerakan seketika");
  Serial.println("  center     -> Parkir kedua servo di tengah (90 derajat)");
  Serial.println("  speed <ms> -> Ubah kecepatan (contoh: speed 3 atau speed 15)");
  Serial.println("  pos <deg>  -> Set sudut manual 0-180 (contoh: pos 45)");
  Serial.println("  status     -> Cek status saat ini");
  Serial.println("  help       -> Tampilkan menu ini lagi");
  Serial.println("=============================================\n");
}

void handleCommand(String cmd) {
  cmd.trim();
  cmd.toLowerCase();

  if (cmd.length() == 0) return;

  if (cmd == "start" || cmd == "s" || cmd == "1") {
    isRunning = true;
    Serial.println(">> [STATE] SERVO DIMULAI (Running...)");
  } 
  else if (cmd == "stop" || cmd == "p" || cmd == "0") {
    isRunning = false;
    Serial.printf(">> [STATE] SERVO DIHENTIKAN di posisi %d derajat.\n", currentAngle);
  } 
  else if (cmd == "center" || cmd == "c") {
    isRunning = false;
    currentAngle = 90;
    servo1.write(90);
    servo2.write(90);
    Serial.println(">> [ACTION] Kedua servo diposisikan ke 90 derajat (Center).");
  } 
  else if (cmd.startsWith("speed")) {
    int newSpeed = cmd.substring(5).toInt();
    if (newSpeed >= 1 && newSpeed <= 100) {
      speedDelayMs = newSpeed;
      Serial.printf(">> [CONFIG] Kecepatan diubah: %d ms per derajat.\n", speedDelayMs);
    } else {
      Serial.println(">> [ERROR] Nilai speed tidak valid (Gunakan rentang 1 s/d 100). Contoh: speed 5");
    }
  } 
  else if (cmd.startsWith("pos")) {
    int targetPos = cmd.substring(3).toInt();
    if (targetPos >= 0 && targetPos <= 180) {
      isRunning = false;
      currentAngle = targetPos;
      servo1.write(targetPos);
      servo2.write(targetPos);
      Serial.printf(">> [ACTION] Kedua servo diarahkan ke %d derajat.\n", targetPos);
    } else {
      Serial.println(">> [ERROR] Sudut tidak valid (Gunakan rentang 0 s/d 180). Contoh: pos 90");
    }
  } 
  else if (cmd == "status") {
    Serial.printf(">> Status: %s | Posisi: %d deg | Speed Delay: %d ms\n", 
                  isRunning ? "RUNNING" : "STOPPED", currentAngle, speedDelayMs);
  } 
  else if (cmd == "help" || cmd == "?") {
    printHelp();
  } 
  else {
    Serial.printf(">> [UNKNOWN] Perintah '%s' tidak dikenali. Ketik 'help' untuk daftar perintah.\n", cmd.c_str());
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // Alokasi timer PWM ESP32 untuk kedua servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);
  ESP32PWM::allocateTimer(3);

  servo1.setPeriodHertz(50);
  servo2.setPeriodHertz(50);

  // Attach Servo 1 ke D18 dan Servo 2 ke D19
  servo1.attach(SERVO1_PIN, 500, 2400);
  servo2.attach(SERVO2_PIN, 500, 2400);

  // Inisialisasi posisi awal ke 0 derajat
  servo1.write(0);
  servo2.write(0);

  printHelp();
  Serial.println("Status Awal: STOPPED. Ketik 'start' di Serial Monitor untuk mulai!");
}

void loop() {
  // 1. Baca input perintah dari Serial Monitor secara non-blocking
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    handleCommand(input);
  }

  // 2. Mesin penggerak non-blocking berbasis millis()
  if (isRunning) {
    unsigned long currentMillis = millis();
    if (currentMillis - lastStepTime >= (unsigned long)speedDelayMs) {
      lastStepTime = currentMillis;

      // Update posisi sudut
      currentAngle += sweepDirection;

      // Balik arah jika menyentuh batas fisik
      if (currentAngle >= 180) {
        currentAngle = 180;
        sweepDirection = -1;
      } else if (currentAngle <= 0) {
        currentAngle = 0;
        sweepDirection = 1;
      }

      // Terapkan ke kedua servo
      servo1.write(currentAngle);
      servo2.write(currentAngle);
    }
  }
}
