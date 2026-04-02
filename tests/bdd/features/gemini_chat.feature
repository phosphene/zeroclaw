Feature: Gemini Provider E2E
  Scenario: Simple chat response
    Given a configured ZeroClaw instance using Gemini
    And Gemini mock is programmed to return "Hello from Gemini"
    When I send the message "Hi"
    Then I should receive the response "Hello from Gemini"

  Scenario: Tool calling flow
    Given a configured ZeroClaw instance using Gemini
    And Gemini mock is programmed to request tool "get_weather" with args '{"location":"London"}'
    And Gemini mock is programmed to then return "The weather in London is 15 degrees"
    When I send the message "What is the weather?"
    Then I should receive the response "The weather in London is 15 degrees"
