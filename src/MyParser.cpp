#include <bits/stdc++.h>

#include "../include/MyParser.h"

Parser::Parser()
{
  //ctor
}

pair<int, vector<double>> Parser::ReadFrom(string path){
  cout << "Start work with: " << path << '\n';
  string line;
  vector<double> out{};
  ifstream f(path);

  getline(f, line);
  int ans = atoi(line.c_str());

  if(f.is_open()){

    while(getline(f, line)) out.push_back(atof(line.c_str()));


  }

  f.close();
  return {ans, out};
}

Parser::~Parser()
{
  //dtor
}
