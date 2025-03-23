"use client"

import "./Translation.css"
import { useState, useEffect, useRef } from "react"
import { ArrowUpDownIcon as ArrowsUpDown, ChevronDown, ChevronUp, Search } from "lucide-react"

const Translation = () => {
  const [sourceText, setSourceText] = useState("")
  const [translatedText, setTranslatedText] = useState("")
  const [sourceLanguage, setSourceLanguage] = useState("auto")
  const [targetLanguage, setTargetLanguage] = useState("vi")
  const [isTranslating, setIsTranslating] = useState(false)
  const [sourceDropdownOpen, setSourceDropdownOpen] = useState(false)
  const [targetDropdownOpen, setTargetDropdownOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const sourceDropdownRef = useRef(null)
  const targetDropdownRef = useRef(null)

  const languages = [
    { code: "auto", name: "Tự động phát hiện", englishName: "Auto detect" },
    { code: "vi", name: "Tiếng Việt", englishName: "Vietnamese" },
    { code: "en", name: "Tiếng Anh", englishName: "English" },
    // { code: "en-AU", name: "Tiếng Anh (Australia)", englishName: "English (Australia)" },
    // { code: "en-CA", name: "Tiếng Anh (Canada)", englishName: "English (Canada)" },
    // { code: "en-IN", name: "Tiếng Anh (Ấn Độ)", englishName: "English (India)" },
    { code: "fr", name: "Tiếng Pháp", englishName: "French" },
    { code: "de", name: "Tiếng Đức", englishName: "German" },
    { code: "ja", name: "Tiếng Nhật", englishName: "Japanese" },
    { code: "ko", name: "Tiếng Hàn", englishName: "Korean" },
    { code: "zh", name: "Tiếng Trung", englishName: "Chinese" },
    { code: "es", name: "Tiếng Tây Ban Nha", englishName: "Spanish" },
  ]

  const getLanguageName = (code) => {
    const language = languages.find((lang) => lang.code === code)
    return language ? language.name : code
  }

  const handleTranslate = () => {
    if (!sourceText.trim()) return

    setIsTranslating(true)

    // Simulate API call for translation
    setTimeout(() => {
      // Simple mock translations for demo purposes
      const mockTranslations = {
        hello: "Xin chào",
        goodbye: "Tạm biệt",
        "thank you": "Cảm ơn bạn",
        "how are you": "Bạn khỏe không",
        "good morning": "Chào buổi sáng",
      }

      const lowerCaseText = sourceText.toLowerCase()
      setTranslatedText(mockTranslations[lowerCaseText] || "Không thể dịch văn bản này")
      setIsTranslating(false)
    }, 800)
  }

  const swapLanguages = () => {
    if (sourceLanguage === "auto" || targetLanguage === "auto") return

    const temp = sourceLanguage
    setSourceLanguage(targetLanguage)
    setTargetLanguage(temp)

    // Also swap text if there's translated content
    if (translatedText) {
      setSourceText(translatedText)
      setTranslatedText(sourceText)
    }
  }

  // Filter languages based on search query
  const filteredLanguages = languages.filter(
    (lang) =>
      lang.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      lang.englishName.toLowerCase().includes(searchQuery.toLowerCase()),
  )

  // Close dropdowns when clicking outside
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (sourceDropdownRef.current && !sourceDropdownRef.current.contains(event.target)) {
        setSourceDropdownOpen(false)
      }
      if (targetDropdownRef.current && !targetDropdownRef.current.contains(event.target)) {
        setTargetDropdownOpen(false)
      }
    }

    document.addEventListener("mousedown", handleClickOutside)
    return () => {
      document.removeEventListener("mousedown", handleClickOutside)
    }
  }, [])

  // Auto-translate when source text changes (with debounce)
  useEffect(() => {
    if (!sourceText.trim()) {
      setTranslatedText("")
      return
    }

    const timer = setTimeout(() => {
      handleTranslate()
    }, 1000)

    return () => clearTimeout(timer)
  }, [sourceText, sourceLanguage, targetLanguage])

  return (
    <div className="translation-page">
      <div className="translation-container">
        <h1 className="translation-title">Translate</h1>

        <div className="language-settings">
          <div className="language-option-wrapper">
            <div className="user-icon">
              <div className="circle-icon">
                <span>V</span>
              </div>
              <ChevronDown size={16} />
            </div>
          </div>
        </div>

        <div className="translation-box">
          <div className="language-selectors">
            <div className="language-selector" ref={sourceDropdownRef}>
              <div className="selected-language" onClick={() => setSourceDropdownOpen(!sourceDropdownOpen)}>
                <span>{getLanguageName(sourceLanguage)}</span>
                {sourceDropdownOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </div>

              {sourceDropdownOpen && (
                <div className="language-dropdown">
                  <div className="search-container">
                    <Search size={16} className="search-icon" />
                    <input
                      type="text"
                      placeholder="Search..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="search-input"
                    />
                  </div>

                  <div className="language-options">
                    {filteredLanguages.map((lang) => (
                      <div
                        key={lang.code}
                        className={`language-option ${sourceLanguage === lang.code ? "active" : ""}`}
                        onClick={() => {
                          setSourceLanguage(lang.code)
                          setSourceDropdownOpen(false)
                          setSearchQuery("")
                        }}
                      >
                        <div className="language-name-primary">{lang.name}</div>
                        <div className="language-name-secondary">{lang.englishName}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <button className="swap-button" onClick={swapLanguages}>
              <ArrowsUpDown size={16} />
            </button>

            <div className="language-selector" ref={targetDropdownRef}>
              <div className="selected-language" onClick={() => setTargetDropdownOpen(!targetDropdownOpen)}>
                <span>{getLanguageName(targetLanguage)}</span>
                {targetDropdownOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
              </div>

              {targetDropdownOpen && (
                <div className="language-dropdown">
                  <div className="search-container">
                    <Search size={16} className="search-icon" />
                    <input
                      type="text"
                      placeholder="Search..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="search-input"
                    />
                  </div>

                  <div className="language-options">
                    {filteredLanguages
                      .filter((lang) => lang.code !== "auto")
                      .map((lang) => (
                        <div
                          key={lang.code}
                          className={`language-option ${targetLanguage === lang.code ? "active" : ""}`}
                          onClick={() => {
                            setTargetLanguage(lang.code)
                            setTargetDropdownOpen(false)
                            setSearchQuery("")
                          }}
                        >
                          <div className="language-name-primary">{lang.name}</div>
                          <div className="language-name-secondary">{lang.englishName}</div>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="translation-content">
            <div className="input-area">
              <textarea
                value={sourceText}
                onChange={(e) => setSourceText(e.target.value)}
                placeholder="Nhập văn bản để dịch..."
                className="translation-input"
              />
            </div>

            <div className="output-area">
              {isTranslating ? (
                <div className="translating-indicator">Đang dịch...</div>
              ) : (
                translatedText && <div className="translated-text">{translatedText}</div>
              )}
            </div>
          </div>

          <button
            className="translate-button"
            onClick={handleTranslate}
            disabled={!sourceText.trim() || isTranslating}
          >
            Dịch
          </button>
        </div>
      </div>
    </div>
  )
}

export default Translation

