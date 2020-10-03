#include "ParallelShrink.cuh"

#include "Timer.h"

// std::vector<Cell<2>> ParallelShrinker::shrink_parallel2(DataSet2& data_set2)
// {
// 	std::vector<Cell<2>> cells_l0{ {0, 0, false} };
//
// 	std::vector<Cell<2>> cells_l1, cells_l2, cells_l3, cells_l4, cells_l5, cells_l6, cells_l7;
// 	Timer::start1();
// 	if (data_set2.layer >= 1)
// 		cells_l1 = process(expand_cells3(cells_l0, data_set2.pt1));
// 	if (data_set2.layer >= 2)
// 		cells_l2 = process(expand_cells3(cells_l1, data_set2.pt2));
// 	if (data_set2.layer >= 3)
// 		cells_l3 = process(expand_cells3(cells_l2, data_set2.pt3));
// 	if (data_set2.layer >= 4)
// 		cells_l4 = process(expand_cells3(cells_l3, data_set2.pt4));
// 	if (data_set2.layer >= 5)
// 		cells_l5 = process(expand_cells3(cells_l4, data_set2.pt5));
// 	if (data_set2.layer >= 6)
// 		cells_l6 = process(expand_cells3(cells_l5, data_set2.pt6));
// 	if (data_set2.layer >= 7)
// 		cells_l7 = process(expand_cells3(cells_l6, data_set2.pt7));
//
// 	Timer::stop1();
// 	switch (data_set2.layer)
// 	{
// 	case 1: return cells_l1;
// 	case 2: return cells_l2;
// 	case 3: return cells_l3;
// 	case 4: return cells_l4;
// 	case 5: return cells_l5;
// 	case 6: return cells_l6;
// 	case 7: return cells_l7;
// 	}
// 	return cells_l0;
// }

std::vector<Cell<3>> ParallelShrinker::shrink_parallel3(DataSet3& data_set3)
{
	std::vector<Cell<3>> cells_l0{ {0, 0, 0, false} };

	std::vector<Cell<3>> cells_l1, cells_l2, cells_l3, cells_l4, cells_l5, cells_l6, cells_l7;
	Timer::start1();
	if (data_set3.layer >= 1)
		cells_l1 = process(expand_cells3(cells_l0, data_set3.pt1));
	if (data_set3.layer >= 2)
		cells_l2 = process(expand_cells3(cells_l1, data_set3.pt2));
	if (data_set3.layer >= 3)
		cells_l3 = process(expand_cells3(cells_l2, data_set3.pt3));
	if (data_set3.layer >= 4)
		cells_l4 = process(expand_cells3(cells_l3, data_set3.pt4));
	if (data_set3.layer >= 5)
		cells_l5 = process(expand_cells3(cells_l4, data_set3.pt5));
	if (data_set3.layer >= 6)
		cells_l6 = process(expand_cells3(cells_l5, data_set3.pt6));
	if (data_set3.layer >= 7)
		cells_l7 = process(expand_cells3(cells_l6, data_set3.pt7));

	Timer::stop1();
	switch (data_set3.layer)
	{
	case 1: return cells_l1;
	case 2: return cells_l2;
	case 3: return cells_l3;
	case 4: return cells_l4;
	case 5: return cells_l5;
	case 6: return cells_l6;
	case 7: return cells_l7;
	}
	return cells_l0;
}

void testParallel2()
{
	// std::vector<Cell<2>> cells{
	// 	{0, 2, false}, {0, 3, true}, {0, 4, false}, {0, 5, true}, {0, 6, true}, {0, 7, true}, {1, 2, false},
	// 	{1, 3, false}, {1, 4, false}, {1, 5, false}, {1, 6, false}, {1, 7, false}, {2, 2, true}, {2, 3, true},
	// 	{3, 2, true}, {3, 3, false}, {4, 0, false}, {4, 1, true}, {4, 2, false}, {4, 3, false}, {5, 0, true},
	// 	{5, 1, false}, {5, 2, false}, {5, 3, false}, {6, 0, false}, {6, 1, false}, {7, 0, true},
	// 	{7, 1, false}
	// };
	//
	ParallelShrinker ps;
	// std::vector<Cell<2>> cells2 = ps.process(cells);
	// std::copy(cells2.begin(), cells2.end(), std::ostream_iterator<Cell<2>>(std::cout, " "));

	std::cout << std::endl;
	// std::vector<Cell<3>> cells3{
	// 	{0, 0, 2, false}, {0, 0, 3, true}, {0, 0, 4, false}, {0, 0, 5, true}, {0, 0, 6, true}, {0, 0, 7, true}, {0, 1, 2, false},
	// 	{0, 1, 3, false}, {0, 1, 4, false}, {0, 1, 5, false}, {0, 1, 6, false}, {0, 1, 7, false}, {0, 2, 2, true}, {0, 2, 3, true},
	// 	{0, 3, 2, true}, {0, 3, 3, false}, {0, 4, 0, false}, {0, 4, 1, true}, {0, 4, 2, false}, {0, 4, 3, false}, {0, 5, 0, true},
	// 	{0, 5, 1, false}, {0, 5, 2, false}, {0, 5, 3, false}, {0, 6, 0, false}, {0, 6, 1, false}, {0, 7, 0, true},
	// 	{0, 7, 1, false}
	// };
	// std::vector<Cell<3>> cells3{
	// 	Cell<3>{0, 0, 2, false}, Cell<3>{0, 0, 3, true}, Cell<3>{0, 0, 4, false}
	// };
	//
	std::vector<Cell<3>> cells3{
		{0, 2, 0, true}, {0, 3, 0, false}, {1, 2, 0, true}, {1, 3, 0, false}, {2, 0, 0, true}, {2, 1, 0, false}, {2, 2, 0, false}, {2, 3, 0, false}, {3, 0, 0, false}, {3, 1, 0, false}, {3, 2, 0, false}, {3, 3, 0, false},
		{0, 2, 1, true}, {0, 3, 1, false}, {1, 2, 1, true}, {1, 3, 1, false}, {2, 0, 1, true}, {2, 1, 1, false}, {2, 2, 1, false}, {2, 3, 1, false}, {3, 0, 1, false}, {3, 1, 1, false}, {3, 2, 1, false}, {3, 3, 1, false},
		{0, 0, 2, true}, {0, 1, 2, false}, {0, 2, 2, false}, {0, 3, 2, false}, {1, 0, 2, true}, {1, 1, 2, false}, {1, 2, 2, false}, {1, 3, 2, true}, {2, 0, 2, false}, {2, 1, 2, false}, {3, 0, 2, false}, {3, 1, 2, true},
		{0, 0, 3, true}, {0, 1, 3, false}, {0, 2, 3, false}, {0, 3, 3, false}, {1, 0, 3, true}, {1, 1, 3, false}, {1, 2, 3, false}, {1, 3, 3, true}, {2, 0, 3, false}, {2, 1, 3, false}, {3, 0, 3, false}, {3, 1, 3, true},

		{2,2,2, true}, {2, 3, 2, true}, {3, 2, 2, true}, {3, 3, 2, false},
		{2,2,3, true}, {2, 3, 3, true}, {3, 2, 3, true}, {3, 3, 3, false}
	};
	std::vector<Cell<3>> cell3a = ps.process(cells3);
	std::copy(cell3a.begin(), cell3a.end(), std::ostream_iterator<Cell<3>>(std::cout, " "));

}
