﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace CLIENT_wpf.WINDOWS
{
    /// <summary>
    /// MakingWindow.xaml에 대한 상호 작용 논리
    /// </summary>
    public partial class MakingWindow : Window
    {
        public DataGetEventHandler DataSendEvent;
        public MakingWindow()
        {
            InitializeComponent();
        }

        private void Click_No(object sender, RoutedEventArgs e)
        {
            
            Window.GetWindow(this).Close();
        }

        private void Click_Yes(object sender, RoutedEventArgs e)
        {
            String temp = "@," + TB_TITLE.Text + "," + TB_PASSWD.Text + "\n";
            Console.WriteLine(temp);
            DataSendEvent(temp);

            Window.GetWindow(this).Close();
        }
        public void Recv_From_Parent(string param)
        {
            Console.WriteLine("Set Action value called");   
        }
    }
}
