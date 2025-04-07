#ifndef MYPARSER_H
#define MYPARSER_H

#include <string>
#include <fstream>
#include <vector>
using namespace std;

class Parser
{
  public:
    Parser();
    pair<int, vector<double>> ReadFrom(string path);
    virtual ~Parser();

  protected:

  private:
};

#endif // PARSER_H
