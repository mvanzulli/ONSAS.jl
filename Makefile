.PHONY: all, update, instantiate, tests, clean, pages, format-check, format, install-hooks

# Julia command
JULIA = julia +1.13
PKG_NAME = ONSAS

# Update dependecies
update:
	$(JULIA) --project=. -e 'import Pkg; Pkg.update()' && \
	$(JULIA) --project=./docs/ -e 'import Pkg; Pkg.update()' && \
	$(JULIA) --project=./test/ -e 'import Pkg; Pkg.update()'

# Update dependecies
instantiate:
	$(JULIA) --project=. -e 'import Pkg; Pkg.instantiate()' && \
	$(JULIA) --project=./docs/ -e 'import Pkg; Pkg.instantiate()' && \
	$(JULIA) --project=./test/ -e 'import Pkg; Pkg.instantiate()'

# Run all tests
tests:
	$(JULIA) --project=. -e 'import Pkg; Pkg.test(;coverage=false, julia_args=["--check-bounds=yes", "--compiled-modules=yes"], force_latest_compatible_version=false, allow_reresolve=true)'

# Make docs
pages:
	$(JULIA) --project=./docs/ -e 'import Pkg; Pkg.instantiate(); include("./docs/make.jl")'

# Remove generated files
clean:
	find . -name "*.cov" -type f -exec rm -f {} +
	find . -name "*.msh" -type f -exec rm -f {} +
	find . -name "*.vti" -type f -exec rm -f {} +
	find . -name "*.vtu" -type f -exec rm -f {} +
	find . -name "*.pvd" -type f -exec rm -f {} +

# Check JuliaFormatter is applied
format-check:
	$(JULIA) --project=. -e '\
	using JuliaFormatter; \
	files = read(`find . -name "*.jl"`, String); \
	for file in split(files, "\n"); \
		if file != "" && !isdir(file) && !format(file, verbose=true, overwrite=false); \
			println("Formatting issue found in: ", file); \
			exit(1); \
		end; \
	end; \
	println("No Formatting issues found.")'

# Apply JuliaFormatter is applied
format:
	$(JULIA) --project=. -e 'using JuliaFormatter; format(".", overwrite=true, verbose=true)'

# Install git hooks
install-hooks:
	cp .githooks/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "Git hooks installed."

# Run all tests
all: tests pages format-check clean
