// neopixel, pushButton control code

#include <Adafruit_NeoPixel.h>

#define ledPin 6        // NeoPixel data pin
#define ledNum 8        // Number of NeoPixel LEDs
#define pushButton 7    // Push button pin

Adafruit_NeoPixel pixels(ledNum, ledPin, NEO_GRB + NEO_KHZ800);

void setup() {
  pinMode(pushButton, INPUT);  // Set push button as input
  pixels.begin();              // Initialize NeoPixel
  pixels.clear();              // Turn off all NeoPixels
  pixels.show();               // Apply changes
  Serial.begin(9600);          // Start serial communication
}

void toggleLights(bool state) {
  if (state) {
    // Turn on NeoPixel (set to white)
    for (int i = 0; i < ledNum; i++) {
      pixels.setPixelColor(i, pixels.Color(128, 128, 128)); // Set to white
    }
  } else {
    // Turn off NeoPixel
    pixels.clear();
  }
  pixels.show();  // Apply changes
}

void loop() {
  // Handle serial commands
  if (Serial.available() > 0) {
    char command = Serial.read();  // Read command from serial
    if (command == '1') {          // If '1' command is received, turn on NeoPixel
      toggleLights(true);          // Turn on NeoPixel
    } else if (command == '0') {   // If '0' command is received, turn off NeoPixel
      toggleLights(false);         // Turn off NeoPixel
    }
  }

  // Check push button state
  int buttonState = digitalRead(pushButton);
  if (buttonState == HIGH) {          // If button is pressed, HIGH
    Serial.println(2);                // Output 2
  } else {
    Serial.println(3);                // If button is not pressed, output 3
  }

  delay(250);  // Delay for debouncing
}
