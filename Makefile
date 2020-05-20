.PHONY: list
list:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

docs:
	dmd -c -D -o- source/nn.d

clean_tests:
	dub clean && rm -- *.lst .*.lst || true

lint:
	dub lint

tests:
	dub test
	# dub run

edit:
	docker-compose -f vim-dev.yml run vim-dev

profile:
	dub run --config=run --build=profile-gc

run:
	dub run --config=run

build:
	dub build --config=run

build_profile:
	dub build --config=run --build=profile-gc

run-gc: build_profile
	./data_science --DRT-gcopt=profile:1
