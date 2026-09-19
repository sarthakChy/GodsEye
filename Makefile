.PHONY: help dashboard script script-all record test clean-demo

help:
	@echo "GodsEye demo targets:"
	@echo "  make dashboard    - launch the Gradio dashboard"
	@echo "  make script       - print a summary of all demo runs"
	@echo "  make script-one R - print summary of a single run (make script-one R=demo_chopping)"
	@echo "  make record       - record a 60-second demo video (needs X display)"
	@echo "  make test         - run the test suite"

dashboard:
	python demo.py dashboard

script:
	python demo.py script-all

script-one:
	python demo.py script outputs/$(R)

record:
	bash record_demo.sh 60

test:
	python -m pytest temporal/tests -q

clean-demo:
	rm -f godseye_demo.mp4
