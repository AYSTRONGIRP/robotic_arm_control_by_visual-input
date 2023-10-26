#include <Servo.h>

Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;

void setup() {
  Serial.begin(115200);  // Set the baud rate of your Serial Monitor to 9600
  servo1.attach(3);    // Attach servo 1 to pin 9
  servo2.attach(5);   // Attach servo 2 to pin 10
  servo3.attach(9);   // Attach servo 3 to pin 11
  servo4.attach(10);   // Attach servo 4 to pin 12
  servo5.attach(11);   // Attach servo 5 to pin 13
}

void loop() {
  if (Serial.available() >= 5) {  // Check if there are at least 5 characters available in the serial buffer
    String inputString = Serial.readString();
    //Serial.println(inputString);  // Read the incoming data as a string
    for (int i = 0; i < 5; i++) {
      if (inputString.charAt(i) == '1') {
        // If the character is '1', rotate the corresponding servo to 180 degrees
        if (i == 0) servo1.write(180);
        else if (i == 1) servo2.write(180);
        else if (i == 2) servo3.write(180);
        else if (i == 3) servo4.write(180);
        else if (i == 4) servo5.write(180);
      } else if (inputString.charAt(i) == '0') {
        // If the character is '0', close the corresponding servo
        if (i == 0) servo1.write(0);
        else if (i == 1) servo2.write(0);
        else if (i == 2) servo3.write(0);
        else if (i == 3) servo4.write(0);
        else if (i == 4) servo5.write(0);
      }
    }
  }
  //delay(1000);
}
