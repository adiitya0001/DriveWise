import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import VehicleDropdown from "./VehicleDropdown";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function App() {
  // ==================================================
  // State
  // ==================================================

  const [vehicles, setVehicles] = useState({});

  const [brand, setBrand] = useState("");
  const [model, setModel] = useState("");

  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState([]);

  const [loading, setLoading] = useState(false);
  const [vehiclesLoading, setVehiclesLoading] = useState(true);
  const [error, setError] = useState("");
  const [loadingMessage, setLoadingMessage] = useState("");

  const displayModel = model.replaceAll("_", " ");

  // ==================================================
  // Suggested Questions
  // ==================================================

  const suggestedQuestions = [
    {
      label: "Safety Features",
      question: `What safety features does the ${displayModel} have?`,
    },
    {
      label: "Engine Options",
      question: `What engine options does the ${displayModel} have?`,
    },
    {
      label: "ADAS",
      question: `What ADAS features does the ${displayModel} have?`,
    },
    {
      label: "Interior & Comfort",
      question: `Tell me about the interior and comfort features of the ${displayModel}.`,
    },
  ];

  // ==================================================
  // Load Vehicles
  // ==================================================

  useEffect(() => {
    const loadVehicles = async () => {
      try {
        setVehiclesLoading(true);
        setError("");

        const response = await fetch(`${API_URL}/vehicles`);
        const data = await response.json();

        if (!response.ok) {
          throw new Error(
            data.detail || "Could not load available vehicles."
          );
        }

        const availableVehicles = data.vehicles || {};

        setVehicles(availableVehicles);

        const brands = Object.keys(availableVehicles);

        if (brands.length > 0) {
          const firstBrand = brands[0];

          setBrand(firstBrand);

          const models = availableVehicles[firstBrand] || [];

          if (models.length > 0) {
            setModel(models[0]);
          }
        }
      } catch (err) {
        setError(err.message);
      } finally {
        setVehiclesLoading(false);
      }
    };

    loadVehicles();
  }, []);

  // ==================================================
  // Loading Messages
  // ==================================================

  useEffect(() => {
    if (!loading) {
      setLoadingMessage("");
      return;
    }

    const loadingMessages = [
      "Searching the brochure...",
      "Finding relevant information...",
      "Generating your answer...",
    ];

    let index = 0;

    setLoadingMessage(loadingMessages[0]);

    const interval = setInterval(() => {
      index = (index + 1) % loadingMessages.length;
      setLoadingMessage(loadingMessages[index]);
    }, 1800);

    return () => clearInterval(interval);
  }, [loading]);

  // ==================================================
  // Brand Change
  // ==================================================

  const handleBrandChange = (selectedBrand) => {
    setBrand(selectedBrand);

    const models = vehicles[selectedBrand] || [];

    if (models.length > 0) {
      setModel(models[0]);
    } else {
      setModel("");
    }

    // New vehicle = new conversation
    setMessages([]);
    setQuestion("");
    setError("");
  };

  // ==================================================
  // Model Change
  // ==================================================

  const handleModelChange = (selectedModel) => {
    setModel(selectedModel);

    // New vehicle = new conversation
    setMessages([]);
    setQuestion("");
    setError("");
  };

  // ==================================================
  // New Chat
  // ==================================================

  const startNewChat = () => {
    setMessages([]);
    setQuestion("");
    setError("");
  };

  // ==================================================
  // Ask DriveWise
  // ==================================================

  const askDriveWise = async () => {
    if (!brand || !model) {
      setError("Please select a brand and model.");
      return;
    }

    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    const currentQuestion = question.trim();

    setLoading(true);
    setError("");

    try {
      // Recent conversation history
      const recentHistory = messages
        .slice(-3)
        .map((message) => ({
          question: message.question,
          answer: message.answer,
        }));

      console.log("Sending history:", recentHistory);

      const response = await fetch(`${API_URL}/ask`, {
        method: "POST",

        headers: {
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          brand,
          model,
          question: currentQuestion,
          history: recentHistory,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Something went wrong."
        );
      }

      setMessages((previousMessages) => [
        ...previousMessages,
        {
          question: currentQuestion,
          answer: data.answer,
          sources: data.sources || [],
        },
      ]);

      setQuestion("");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // ==================================================
  // UI
  // ==================================================

  return (
    <div className="app">
      <div className="container">

        {/* ==========================================
            Header
        ========================================== */}

        <header className="app-header">
          <div className="header-title">
            <h1>DriveWise</h1>

            <p>
              AI Automotive Brochure Assistant
            </p>
          </div>

          {messages.length > 0 && (
            <button
              type="button"
              className="new-chat-button"
              onClick={startNewChat}
              disabled={loading}
            >
              + New Chat
            </button>
          )}
        </header>

        {/* ==========================================
            Vehicle Selectors
        ========================================== */}

        <div className="selectors">
          <VehicleDropdown
            label="Brand"
            value={brand}
            options={Object.keys(vehicles)}
            onChange={handleBrandChange}
            disabled={vehiclesLoading}
            placeholder={
              vehiclesLoading
                ? "Loading brands..."
                : "Select brand"
            }
          />

          <VehicleDropdown
            label="Model"
            value={model}
            options={vehicles[brand] || []}
            onChange={handleModelChange}
            disabled={vehiclesLoading || !brand}
            placeholder="Select model"
          />
        </div>

        {/* ==========================================
            Question Area
        ========================================== */}

        <div className="question-box">

          {/* Suggested Questions */}

          <div className="suggestions">
            <span className="suggestions-title">
              Try asking
            </span>

            <div className="suggestion-buttons">
              {suggestedQuestions.map((item) => (
                <button
                  key={item.label}
                  type="button"
                  className="suggestion-chip"
                  onClick={() => {
                    setQuestion(item.question);
                    setError("");
                  }}
                  disabled={!model || loading}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>

          {/* Question Input */}

          <textarea
            placeholder={
              model
                ? `Ask anything about the ${displayModel}...`
                : "Select a vehicle first..."
            }
            value={question}
            onChange={(event) => {
              setQuestion(event.target.value);
              setError("");
            }}
            onKeyDown={(event) => {
              if (
                event.key === "Enter" &&
                !event.shiftKey
              ) {
                event.preventDefault();

                if (
                  question.trim() &&
                  !loading &&
                  model
                ) {
                  askDriveWise();
                }
              }
            }}
            disabled={!model || loading}
          />

          {/* Ask Button */}

          <button
            onClick={askDriveWise}
            disabled={
              loading ||
              vehiclesLoading ||
              !brand ||
              !model
            }
          >
            {loading ? (
              <span className="button-loading">
                <span className="spinner"></span>
                Thinking...
              </span>
            ) : (
              "Ask DriveWise"
            )}
          </button>

          {/* Loading Status */}

          {loading && (
            <div className="loading-status">
              <span className="loading-dot"></span>
              {loadingMessage}
            </div>
          )}
        </div>

        {/* ==========================================
            Error
        ========================================== */}

        {error && (
          <div className="error">
            {error}
          </div>
        )}

        {/* ==========================================
            Conversation
        ========================================== */}

        {messages.length > 0 && (
          <div className="conversation">
            {messages.map((message, index) => (
              <div
                className="conversation-item"
                key={index}
              >
                {/* User */}

                <div className="user-message">
                  <div className="message-label">
                    You
                  </div>

                  <div className="user-bubble">
                    {message.question}
                  </div>
                </div>

                {/* DriveWise */}

                <div className="assistant-message">
                  <div className="message-label">
                    DriveWise
                  </div>

                  <div className="assistant-bubble">

                    {/* Answer */}

                    <div className="answer">
                      <ReactMarkdown>
                        {message.answer}
                      </ReactMarkdown>
                    </div>

                    {/* Sources */}

                    {message.sources?.length > 0 && (
                      <details className="sources">
                        <summary>
                          Sources
                        </summary>

                        <div className="source-content">
                          <div>
                            <strong>
                              Official{" "}
                              {brand}{" "}
                              {displayModel}{" "}
                              Brochure
                            </strong>

                            <p className="source-description">
                              Information used to generate this answer
                            </p>
                          </div>

                          <span className="source-pages">
                            Pages{" "}
                            {[
                              ...new Set(
                                message.sources.map(
                                  (source) =>
                                    source.page
                                )
                              ),
                            ]
                              .filter(Boolean)
                              .sort(
                                (a, b) =>
                                  a - b
                              )
                              .join(", ")}
                          </span>
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;