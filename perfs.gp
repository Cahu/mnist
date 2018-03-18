while (1) {
	set xlabel  'Epoch'
	set ylabel  'Accuracy (%)'
	set y2label 'Cost'
	set y2tics
	plot 'results.dat' using 1:2 with lines axes x1y1 title "Accuracy", \
	     'results.dat' using 1:3 with lines axes x1y2 title "Cost"
	pause 1
}
