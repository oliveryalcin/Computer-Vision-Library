#pragma once
#include <iostream>
#include <array>

// Class to manage subsets of the circle (the arcs) and organising combinations efficiently
class Arcs {

private:

	static const int MAX_ARCS = 16; //if every pixel is an alternating series between brighter, darker, similar..
	enum ArcType {
		NA,
		Darker,
		Brighter,
		Similar
	};

	int numOfPoints = 0;
	int numOfArcs = 0;

	typedef struct {
		int startIndex = 0;
		int endIndex = 0;
		ArcType attributeType = NA;
		int len = 0;

	}Arc;
	
	std::array<Arc, MAX_ARCS> arcs;

public:

	Arcs();
	void setArc(int startIdx, int endIdx);






		





};