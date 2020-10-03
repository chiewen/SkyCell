#pragma once
#include <algorithm>
#include <cstdarg>
#include <iterator>
#include <ostream>

template <int E>
class Iterator;

template <int D>
class KeyCell
{
public:
	unsigned short int indices[D];
	KeyCell(std::initializer_list<unsigned short int> il);

	Iterator<D - 1> get_I() const;
	unsigned short int get_last() const;

	unsigned short int& operator[](int i) { return indices[i]; }

	friend std::ostream& operator<<(std::ostream& os, const KeyCell& p)
	{
		os << "[";
		std::copy(std::begin(p.indices), std::end(p.indices), std::ostream_iterator<unsigned short int>(os, " "));
		os << "]";
		return os;
	}

	friend bool operator==(const KeyCell& kc1, const KeyCell& kc2)
	{
		return std::equal(std::begin(kc1.indices), std::end(kc1.indices), std::begin(kc2.indices));
	}

	friend bool operator!=(const KeyCell& kc1, const KeyCell& kc2)
	{
		return !(kc1 == kc2);
	}
};

template <int D>
KeyCell<D>::KeyCell(std::initializer_list<unsigned short int> il)
{
	if (il.size() > 0)
	{
		int i = 0;
		for (auto elem : il)
		{
			indices[i++] = elem;
		}
	}
	else
	{
		memset(indices, 0, sizeof(unsigned short) * D);
	}
}

template <int D>
Iterator<D - 1> KeyCell<D>::get_I() const
{
	return Iterator<D - 1>{(std::initializer_list<unsigned short>(std::begin(indices), std::end(indices) - 1))};
}

template <int D>
unsigned short KeyCell<D>::get_last() const
{
	return indices[D - 1];
}

template <int E>
class Iterator : public KeyCell<E>
{
public:
	Iterator(std::initializer_list<unsigned short int> il);
	void advance(int max);
	Iterator<E> next_layer();

	friend bool operator <(const Iterator<E>& i1, const Iterator<E>& i2)
	{
		for (int i = 0; i < E; ++i)
		{
			if (i1.indices[i] < i2.indices[i])
				return true;
			if (i1.indices[i] > i2.indices[i])
				return false;
		}
		return false;
	}
};

template <int E>
Iterator<E>::Iterator(std::initializer_list<unsigned short int> il) : KeyCell(il)
{
}

template <int E>
void Iterator<E>::advance(int max)
{
	int d = E - 1;
	while (KeyCell<E>::indices[d] == max - 1)
	{
		_ASSERT(d >= 0);
		KeyCell<E>::indices[d--] = 0;
	}
	++KeyCell<E>::indices[d];
}

template <int E>
Iterator<E> Iterator<E>::next_layer()
{
	Iterator<E> result{};
	for (int i = 0; i < E; i++)
	{
		result.indices[i] = KeyCell<E>::indices[i] * 2;
	}
	return result;
}
