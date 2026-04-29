#include <Servo.h>

// ==========================================
// 1. PIN DEFINITIONS
// ==========================================

// L298N Motor Driver Pins
const int ENA = 4;  
const int IN1 = 22; 
const int IN2 = 24; 
const int IN3 = 26; 
const int IN4 = 28; 
const int ENB = 5;  

// HC-SR04 Ultrasonic Sensor Pins
const int trigPin = 30; 
const int echoPin = 31; 

// Robot State Variables
bool isDriving = false; 
bool isBlocked = false; 
bool isRotating = false; 

// Servo Objects
Servo baseServo;     
Servo shoulderA1;    
Servo shoulderA2;    
Servo elbowServo;    
Servo wristPitch;    
Servo wristRoll;     
Servo gripperServo;  

// --- CONSTANTS FOR TUNING ---
const int HOME_SHOULDER = 110;  // Elevated position (start position)
const int HOME_ELBOW = 150;     // Tucked safely
const int PICKUP_LOW = 70;      // Lying flat (for knocking down and picking up)

// ==========================================
// 2. SETUP
// ==========================================

void setup() {
  Serial.begin(115200); 
  Serial.println("Robot Initializing...");

  pinMode(ENA, OUTPUT);
  pinMode(ENB, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  stopMotors();

  baseServo.attach(12);
  shoulderA1.attach(11);
  shoulderA2.attach(10);
  elbowServo.attach(9);
  wristPitch.attach(8);
  wristRoll.attach(7);
  gripperServo.attach(6);

  setArmHome();
  
  Serial.println("Robot Ready. Send F, S, A, or D.");
}

// ==========================================
// 3. MAIN LOOP 
// ==========================================

void loop() {
  int distance = getDistance();

  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == 'F') {
      isDriving = true; 
      isRotating = false;
      Serial.println("Driving Forward...");
    } 
    else if (command == 'S') {
      isDriving = false; 
      isBlocked = false; 
      isRotating = false;
      stopMotors();
      Serial.println("Stopped.");
    }
    else if (command == 'L') {
      isDriving = false; 
      isRotating = true;
      driveMotors(-150, 150); // Turn Left
      Serial.println("Turning Left...");
    }
    else if (command == 'R') {
      isDriving = false; 
      isRotating = true;
      driveMotors(150, -150); // Turn Right
      Serial.println("Turning Right...");
    }
    else if (command == 'A') {
      isDriving = false;
      isBlocked = false;
      isRotating = false;
      stopMotors(); 
      Serial.println("Executing Double-Lowering Pick-Up Sequence");

      // 1. Prepare joints (Base, Arm, Wrist, Gripper)
      baseServo.write(90);      // Base center
      elbowServo.write(180-90); // Elbow center
      wristPitch.write(90);     // Wrist level
      wristRoll.write(90);      // Wrist level
      gripperServo.write(45);   // Gripper Open
      delay(500);

      // FIRST LOWERING (To knock down)
      moveShoulder(HOME_SHOULDER, PICKUP_LOW); // Lower Arm
      delay(500);

      // RAISE ARM
      moveShoulder(PICKUP_LOW, HOME_SHOULDER); 
      delay(500);

      // SECOND LOWERING (To grab)
      moveShoulder(HOME_SHOULDER, PICKUP_LOW);
      delay(800);
      
      gripperServo.write(115); // Gripper Close (Grab)
      delay(1000);
      
      // LIFT ARM
      moveShoulder(PICKUP_LOW, HOME_SHOULDER);
      delay(500);
      
      Serial.println("DONE"); // Signal Python that action is finished
    }
    else if (command == 'D') {
      isDriving = false;
      isBlocked = false;
      stopMotors();
      Serial.println("Executing Drop and Rotate Sequence");

      // Lower Arm to drop position
      moveShoulder(HOME_SHOULDER, PICKUP_LOW);
      delay(500);

      // Open Gripper
      gripperServo.write(45); 
      delay(1000);

      // Return Arm to home
      setArmHome(); 
      delay(500);

      // Rotate 180 degrees (Go back to search area)
      isRotating = true;
      Serial.println("Rotating 180 degrees...");
      driveMotors(150, -150); 
      delay(1200); 
      stopMotors();
      isRotating = false;
      
      Serial.println("DONE"); // Signal Python that action is finished
    }
  }

  // SMART CRUISE CONTROL
  if (isDriving) {
    if (distance > 0 && distance < 12) {
      stopMotors(); 
      if (!isBlocked) {
        Serial.println("Obstacle! Pausing...");
        isBlocked = true; 
      }
    } else {
      driveMotors(150, 150); 
      if (isBlocked) {
        Serial.println("Path clear! Resuming...");
        isBlocked = false;
      }
    }
  }
  delay(50);
}

// 4. MOTOR CONTROL

void driveMotors(int leftSpeed, int rightSpeed) {
  digitalWrite(IN1, leftSpeed > 0 ? HIGH : LOW);
  digitalWrite(IN2, leftSpeed > 0 ? LOW : HIGH);
  digitalWrite(IN3, rightSpeed > 0 ? HIGH : LOW);
  digitalWrite(IN4, rightSpeed > 0 ? LOW : HIGH);
  analogWrite(ENA, abs(leftSpeed));
  analogWrite(ENB, abs(rightSpeed));
}

void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  analogWrite(ENA, 0); analogWrite(ENB, 0);
}

// 5. ARM CONTROL

void moveShoulder(int start_pos, int end_pos) {
  if (start_pos < end_pos) {
    for(int pos = start_pos; pos <= end_pos; pos++) {
      shoulderA1.write(pos);
      shoulderA2.write(180 - pos);
      delay(20); 
    }
  } else {
    for(int pos = start_pos; pos >= end_pos; pos--) {
      shoulderA1.write(pos);
      shoulderA2.write(180 - pos);
      delay(20); 
    }
  }
}

void moveArmAngles(int base, int shoulder, int elbow, int wPitch, int wRoll, int grip) {
  baseServo.write(constrain(base, 0, 180));
  shoulderA1.write(constrain(shoulder, 0, 180));
  shoulderA2.write(180 - constrain(shoulder, 0, 180));
  elbowServo.write(180 - constrain(elbow, 0, 180));
  wristPitch.write(constrain(wPitch, 0, 180));
  wristRoll.write(constrain(wRoll, 0, 180));
  gripperServo.write(constrain(grip, 33, 115));
}

void setArmHome() {
  int current = shoulderA1.read();
  
  // Smoothly transition to the correct elevation
  moveShoulder(current, HOME_SHOULDER);

  // Final tucked position
  moveArmAngles(90, HOME_SHOULDER, HOME_ELBOW, 90, 90, 90);
}

// 6. SENSOR

int getDistance() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  long duration = pulseIn(echoPin, HIGH, 25000); 
  return duration * 0.034 / 2;
}