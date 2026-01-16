.PHONY: help build run stop clean deploy test

help:
	@echo "Доступные команды:"
	@echo "  build     - Сборка Docker образа"
	@echo "  run       - Запуск приложения в Docker"
	@echo "  stop      - Остановка приложения"
	@echo "  clean     - Очистка Docker образов и контейнеров"
	@echo "  test      - Запуск тестов"
	@echo "  deploy    - Деплой на сервер"

build:
	docker-compose build

run:
	docker-compose up -d

stop:
	docker-compose down

clean:
	docker-compose down -v
	docker system prune -f

test:
	docker-compose run --rm app python -m pytest tests/ -v

deploy: build
	@echo "Запуск деплоя..."
	# Добавьте команды для деплоя