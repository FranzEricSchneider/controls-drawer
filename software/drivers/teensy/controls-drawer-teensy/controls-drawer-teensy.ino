/*
  Blink
  Turns on an LED on for one second, then off for one second, repeatedly.
 
  This example code is in the public domain.
 */
int limit_x_low = 9;
int limit_x_high = 10;
int limit_y_low = 11;
int limit_y_high = 12;
int led = 13;

// the setup routine runs once when you press reset:
void setup() {                
  // initialize the digital pin as an output.
  pinMode(led, OUTPUT);
  // Start serial communication
  Serial.begin(9600);
}

void printState(int x_low, int x_high, int y_low, int y_high) {
    Serial.print("x_low,");
    Serial.print(x_low);
    Serial.print(",x_high,");
    Serial.print(x_high);
    Serial.print(",y_low,");
    Serial.print(y_low);
    Serial.print(",y_high,");
    Serial.print(y_high);
    Serial.print("\n");
}

// the loop routine runs over and over again forever:
void loop() {
  digitalWrite(led, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(1000);               // wait for a second
  digitalWrite(led, LOW);    // turn the LED off by making the voltage LOW
  delay(1000);               // wait for a second
  printState(digitalRead(limit_x_low),
             digitalRead(limit_x_high),
             digitalRead(limit_y_low),
             digitalRead(limit_y_high));
}
