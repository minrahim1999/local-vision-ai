# Contributing to Local Vision AI

Thank you for considering contributing! This project is designed to run entirely on-device across multiple platforms.

## How to Contribute

### Reporting Issues

When opening an issue, please include:
- Your platform (macOS / Windows / Linux)
- Hardware (Apple Silicon M1/M2/M3/M4, NVIDIA GPU model, or CPU)
- Python version (`python --version`)
- The output of `make audit`
- Steps to reproduce

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Make your changes
4. Run tests: `make test`
5. Ensure code passes linting: `uv run ruff check .`
6. Commit with a clear message
7. Push and open a PR

## Code Standards

- **Python version:** 3.10+
- **Type hints:** Required on all public functions
- **Error handling:** No bare `except:` — catch specific exceptions
- **No hard-coded paths:** Use environment variables or config files
- **No secrets:** Never commit API keys, tokens, or absolute paths
- **Logging:** Use `logging.getLogger(__name__)` instead of `print()`

## Adding a New Backend

See `docs/backends.md` for the full guide. The short version:

1. Create `services/backends/your_backend.py`
2. Inherit from `TextToImageBackend` or `ImageToTextBackend`
3. Implement all abstract methods
4. Add to `T2I_BACKENDS` or `I2T_BACKENDS` in `factory.py`
5. Add config section to `config/text_to_image.yaml` or `config/image_to_text.yaml`
6. Add tests
7. Update `docs/backends.md`

## Testing

```bash
# Run all tests
make test

# Run specific test file
uv run pytest tests/test_schemas.py -v

# Run with timeout
uv run pytest tests/ -v --timeout=60
```

## Commit Message Style

- Use present tense: "Add feature" not "Added feature"
- Use imperative mood: "Fix bug" not "Fixes bug"
- Reference issues if applicable: "Fix memory leak (#123)"
- No AI co-author attribution

## License

By contributing, you agree that your contributions will be licensed under the same license as the project. See `LICENSE` for details.

---

*This project uses the existing Git identity. Never add automated co-author attribution to commits.*
