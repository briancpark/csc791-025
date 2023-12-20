# Makefile for LaTeX project

# List of source LaTeX files
SOURCES := $(wildcard *.tex)
PDFS := $(SOURCES:.tex=.pdf)

# Default target
all: $(PDFS)

# Rule to compile LaTeX files
%.pdf: %.tex
	pdflatex -shell-escape $<
	-bibtex $(<:.tex=) 2>/dev/null
	pdflatex -shell-escape $<
	pdflatex -shell-escape $<

# Clean up temporary files
clean:
	rm -f *.aux *.log *.out *.toc *.bak* *pdf *.bbl *.blg *.synctex.gz
	rm -rf *_minted-*

# Lint .tex files
lint:
	for file in $(SOURCES); do latexindent -w -s $$file; done

# Check if .tex files are properly formatted
check:
	@errors=0; \
	for file in $(SOURCES); do \
		latexindent $$file > $$file.formatted; \
		if ! diff -q $$file $$file.formatted > /dev/null; then \
			echo "Formatting issue detected in $$file"; \
			errors=$$((errors + 1)); \
		fi; \
		rm -f $$file.formatted; \
	done; \
	if [ $$errors -ne 0 ]; then \
		echo "Formatting issues found in $$errors files."; \
		exit 1; \
	else \
		echo "All files are properly formatted."; \
	fi

# Help
help:
	@echo "Available commands:"
	@echo "  make           Compile all LaTeX files to PDF"
	@echo "  make clean     Remove temporary files"
	@echo "  make lint      Format .tex files using latexindent"
	@echo "  make check     Check if .tex files are properly formatted"