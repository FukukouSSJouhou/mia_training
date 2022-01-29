import QtQuick 2.12
import QtQuick.Window 2.12
import QtQuick.Controls 2.12
import QtCharts 2.3

Window {
    id:rootwindow
    width: 640
    height: 480
    visible: true
    title: qsTr("Mia Training Application")

    ChartView {
        id: chart
        width: rootwindow.width
        height: rootwindow.height*0.7
        anchors.horizontalCenter: parent.horizontalCenter
        anchors.verticalCenter: parent.verticalCenter
        anchors.verticalCenterOffset: 67
        anchors.horizontalCenterOffset: 0

        ValueAxis {
            id: axisX

         }

        ValueAxis{
            id: axisY
            min:0
            max:1.5
        }
        LineSeries {
            name: "Loss"
            axisX: axisX
            axisY:axisY
        }
    }

    Button {
        id: button
        x: 119
        y: 21
        text: qsTr("Button")
        onClicked: {
            mainwinconnect.clickedkun()
        }
    }

    Component.onCompleted: {
        mainwinconnect.value_upd.connect(
                    function tdn(floatkunniki){
                        mainwinconnect.printkun(floatkunniki)
                        axisX.max=floatkunniki
                        //chart.axisX(.)


                    }

        );
        mainwinconnect.setserieskun(chart.series(0),axisX,axisY)
    }
}
