"use client"

import { useState } from "react"
import { ArrowLeftRight } from "lucide-react"

const Translation = () => {
  const [sourceLanguage, setSourceLanguage] = useState("English")
  const [targetLanguage, setTargetLanguage] = useState("Spanish")
  const [sourceText, setSourceText] = useState("")
  const [translatedText, setTranslatedText] = useState("")

  const languages = [
    "English",
    "Spanish",
    "French",
    "German",
    "Italian",
    "Portuguese",
    "Russian",
    "Japanese",
    "Chinese",
    "Korean",
  ]

  const handleSwapLanguages = () => {
    setSourceLanguage(targetLanguage)
    setTargetLanguage(sourceLanguage)
    setSourceText(translatedText)
    setTranslatedText(sourceText)
  }

  const handleTranslate = () => {
    // In a real app, this would call a translation API
    setTranslatedText(`[Translated ${sourceLanguage} to ${targetLanguage}]: ${sourceText}`)
  }

  return (
    <div className="page-container">
      <div className="page-header">Translation</div>
      <div className="translation-container">
        <div className="language-selectors">
          <div className="language-selector">
            <div className="select-container">
              <select className="select" value={sourceLanguage} onChange={(e) => setSourceLanguage(e.target.value)}>
                {languages.map((lang) => (
                  <option key={lang} value={lang}>
                    {lang}
                  </option>
                ))}
              </select>
              <div className="select-icon">▼</div>
            </div>
          </div>

          <button className="swap-button" onClick={handleSwapLanguages}>
            <ArrowLeftRight size={20} />
          </button>

          <div className="language-selector">
            <div className="select-container">
              <select className="select" value={targetLanguage} onChange={(e) => setTargetLanguage(e.target.value)}>
                {languages.map((lang) => (
                  <option key={lang} value={lang}>
                    {lang}
                  </option>
                ))}
              </select>
              <div className="select-icon">▼</div>
            </div>
          </div>
        </div>

        <div className="translation-section">
          <label className="label">Source Text</label>
          <textarea
            className="textarea"
            placeholder="Enter text to translate..."
            value={sourceText}
            onChange={(e) => setSourceText(e.target.value)}
          ></textarea>
        </div>

        <div className="translation-section">
          <label className="label">Translated Text</label>
          <textarea
            className="textarea"
            placeholder="Translation will appear here..."
            value={translatedText}
            readOnly
          ></textarea>
        </div>

        <button className="button" onClick={handleTranslate}>
          Translate
        </button>
      </div>
    </div>
  )
}

export default Translation

