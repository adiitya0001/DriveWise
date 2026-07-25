import { useEffect, useRef, useState } from "react";

function VehicleDropdown({
  label,
  value,
  options,
  onChange,
  disabled = false,
  placeholder = "Select an option",
}) {
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef(null);

  // ==================================================
  // Format vehicle names
  // XUV_7XO -> XUV 7XO
  // creta   -> Creta
  // ==================================================

  const formatName = (name) => {
    if (!name) return "";

    const formatted = name.replaceAll("_", " ");

    return formatted
      .split(" ")
      .map((word) => {
        // Keep words with numbers or already-uppercase names
        if (
          /\d/.test(word) ||
          word === word.toUpperCase()
        ) {
          return word;
        }

        return (
          word.charAt(0).toUpperCase() +
          word.slice(1)
        );
      })
      .join(" ");
  };

  // ==================================================
  // Close when clicking outside
  // ==================================================

  useEffect(() => {
    const handleOutsideClick = (event) => {
      if (
        dropdownRef.current &&
        !dropdownRef.current.contains(event.target)
      ) {
        setOpen(false);
      }
    };

    document.addEventListener(
      "mousedown",
      handleOutsideClick
    );

    return () => {
      document.removeEventListener(
        "mousedown",
        handleOutsideClick
      );
    };
  }, []);

  // ==================================================
  // Close with Escape
  // ==================================================

  useEffect(() => {
    const handleEscape = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
      }
    };

    document.addEventListener(
      "keydown",
      handleEscape
    );

    return () => {
      document.removeEventListener(
        "keydown",
        handleEscape
      );
    };
  }, []);

  // ==================================================
  // Select option
  // ==================================================

  const handleSelect = (option) => {
    onChange(option);
    setOpen(false);
  };

  // ==================================================
  // UI
  // ==================================================

  return (
    <div
      className={`custom-dropdown ${
        open ? "custom-dropdown-open" : ""
      }`}
      ref={dropdownRef}
    >
      <label className="dropdown-label">
        {label}
      </label>

      {/* Trigger */}

      <button
        type="button"
        className={`dropdown-trigger ${
          open ? "dropdown-trigger-open" : ""
        }`}
        onClick={() => {
          if (!disabled) {
            setOpen((current) => !current);
          }
        }}
        disabled={disabled}
        aria-expanded={open}
        aria-haspopup="listbox"
      >
        <span
          className={
            value
              ? "dropdown-value"
              : "dropdown-placeholder"
          }
        >
          {value
            ? formatName(value)
            : placeholder}
        </span>

        {/* Chevron */}

        <span
          className={`dropdown-chevron ${
            open ? "dropdown-chevron-open" : ""
          }`}
        >
          <svg
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="m6 9 6 6 6-6" />
          </svg>
        </span>
      </button>

      {/* Dropdown Menu */}

      {open && !disabled && (
        <div
          className="dropdown-menu"
          role="listbox"
        >
          {options.map((option) => {
            const selected = option === value;

            return (
              <button
                key={option}
                type="button"
                role="option"
                aria-selected={selected}
                className={`dropdown-option ${
                  selected
                    ? "dropdown-option-selected"
                    : ""
                }`}
                onClick={() =>
                  handleSelect(option)
                }
              >
                <span>
                  {formatName(option)}
                </span>

                {selected && (
                  <span
                    className="dropdown-check"
                    aria-hidden="true"
                  >
                    ✓
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default VehicleDropdown;