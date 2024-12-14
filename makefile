export_dependencies:
	poetry export -f requirements.txt --without-hashes | sed 's/;.*$//' > requirements.txt